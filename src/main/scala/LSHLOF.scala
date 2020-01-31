package org.apache.spark.ml.outlier

import org.apache.log4j.LogManager

import org.apache.spark.ml.outlier.utils._

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DatasetUtils, Identifiable, SchemaUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

import scala.math.Ordering

private[outlier] trait LOFParams
    extends Params
    with HasFeaturesCol
    with HasOutputCol
    with HasPredictionCol {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    //SchemaUtils.validateVectorCompatibleColumn(schema, getFeaturesCol)
    SchemaUtils.checkColumnType(schema, $ { featuresCol }, VectorType)

    StructType(
      Array(
        StructField(LSHLOF.index, DataTypes.LongType, false),
        StructField(LSHLOF.lof, DataTypes.DoubleType, false),
        StructField($(predictionCol), DataTypes.DoubleType, false)
      )
    )

  }

  /**
    * The number of nearest neighbors as in k-nearest neighbors
    * @group param
    */
  final val numNeighbors: IntParam =
    new IntParam(this, "numNeighbors", "", ParamValidators.gt(0))

  /** @group getParam */
  def getNumNeighbors: Int = $(numNeighbors)

  /**
    * The percentage of target outliers to be detected among the entire dataset.
    * @group param
    */
  final val contamination: DoubleParam =
    new DoubleParam(
      this,
      "contamination",
      "the proportion of outliers in the data set. in range(0, 1]",
      ParamValidators.inRange(0, 1, false, true)
    )

  /** @group getParam */
  def getContamination: Double = $(contamination)

  /**
    * The extension percentage for extra outlier candidates, used in cross-partition updating.
    * The size of candidates is: dataSize * math.min(($(contaminExtend) + $(contamination)), 1.0)
    * @group param
    */
  final val contaminExtend: DoubleParam =
    new DoubleParam(
      this,
      "contaminExtend",
      "",
      ParamValidators.inRange(0, 1, false, false)
    )

  /** @group getParam */
  def getContaminExtend: Double = $(contaminExtend)

  /**
    * The number of data partitions for the Spark RDD.
    * @group param
    */
  final val numPartitions: IntParam =
    new IntParam(this, "numPartitions", "", ParamValidators.gt(0))

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /** below is LSH-related parameters*/
  /**
    * The number of hash functions for LSH.
    * @group param
    */
  final val numFunctions: IntParam =
    new IntParam(this, "numFunctions", "", ParamValidators.gt(0))

  /** @group getParam */
  def getNumFunctions: Int = $(numFunctions)

  /**
    * The bucket length (width) for each LSH hash function.
    * @group param
    */
  final val w: DoubleParam =
    new DoubleParam(this, "w", "", ParamValidators.gt(0))

  /** @group getParam */
  def getW: Double = $(w)
}

/**
  * @param uid unique ID for Transformer
  * */
class LSHLOF(override val uid: String)
    extends Transformer
    with LOFParams
    with HasSeed {

  def this() = this(Identifiable.randomUID("LSHLOF"))

  // Set default values upon the creation of the object.
  // The default values can be overriden by using setXX methods
  setDefault(
    numNeighbors -> 30,
    contamination -> 0.05,
    contaminExtend -> 0.05,
    numPartitions -> 10,
    numFunctions -> 10,
    w -> 4.0,
    seed -> this.getClass.getName.hashCode.toLong
  )

  /** @group setParam */
  def setNumNeighbors(value: Int): this.type = set(numNeighbors, value)

  /** @group setParam */
  def setContamination(value: Double): this.type = set(contamination, value)

  /** @group setParam */
  def setContaminExtend(value: Double): this.type = set(contaminExtend, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setNumFunctions(value: Int): this.type = set(numFunctions, value)

  /** @group setParam */
  def setW(value: Double): this.type = set(w, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /**
    * Identify the outliers from a dataset.
    *
    * @param dataset input dataset. The dataset must have a feature column named "features".
    *                Or one can use setFeaturesCol of LSHLOF to set the feature column name.
    *                The feature column is VectorType.
    *                One can use VectorAssembler to merge multiple columns into a feature column.
    *
    * @return the outlier candidates (contamination + comtaminExtend)
    *         in the form of Tuple(index, lofScore, isOutlier)
    * */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val schema = transformSchema(dataset.schema, logging = true)
    val session = dataset.sparkSession

    val instanceRdd = columnToVector(dataset, getFeaturesCol)
    if (dataset.storageLevel == StorageLevel.NONE) {
      instanceRdd.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val dataSize = instanceRdd.count()

    val indexedRdd = instanceRdd.zipWithIndex().map(_.swap)
    val lofRdd = distributedLOF(indexedRdd).cache()
    //(ID, (vect, kDistance, density, lof))

    val outlierCandidatesRdd = getOutlierCandiates(lofRdd)

    val candidateScoreRdd =
      crossparitionUpdating(outlierCandidatesRdd, lofRdd)

    val outlierSize = dataSize * $(contamination)
    val finalCandidateScoreRdd = candidateScoreRdd
      .zipWithIndex()
      .map {
        case ((indx, lof), tempIndx) =>
          val isOutlier = if (tempIndx < outlierSize) 1.0 else 0.0
          Row(indx, lof, isOutlier)
      }
    session.createDataFrame(finalCandidateScoreRdd, schema)
  }

  /**
    * Send the outlier candidates to each partition, based on the
    * computations from which, update the LOF scores of the candidates
    *
    * @param candidateRdd RDD of outlier candidates. Each element in the form of
    *                     Tuple(Index, (vector, kDistance, density, lof))
    * @param lofRdd RDD of the entire dataset with lof scores. Each element in the form of
    *               Tuple(Index, (vector, kDistance, density, lof))
    *
    * @return final outlier scores for the candidates. Tuple(index, lof)
    * */
  private def crossparitionUpdating(
    candidateRdd: RDD[(Long, (Vector, Double, Double, Double))],
    lofRdd: RDD[(Long, (Vector, Double, Double, Double))]
  ) = {
    val sc = candidateRdd.sparkContext

    val bcCandidates = sc.broadcast(candidateRdd.collect())

    val candidateNeighborhoodRdd = lofRdd
      .mapPartitions {
        case iter =>
          val partitionPointArray = iter.toArray
          val actualNumNeighbors =
            math.min($(numNeighbors), partitionPointArray.size - 1)
          val broadcastCandidates = bcCandidates.value
          broadcastCandidates.map {
            case (indx, (vect, _, _, _)) =>
              (
                indx,
                computeNeighborhood(
                  indx,
                  vect,
                  partitionPointArray,
                  actualNumNeighbors
                )
              )
          }.toIterator
      }
      .reduceByKey {
        case (first, second) =>
          mergeNeigbhorhood(first, second, $(numNeighbors))
      }

    val finalCandiateScores = candidateNeighborhoodRdd
      .map {
        case (indx, neigbhorhood) =>
          val neighborAverageDensity = neigbhorhood.map { _._2._3 }.sum / neigbhorhood.size
          val lrd = neigbhorhood.size / neigbhorhood.map { x =>
            scala.math.max(x._2._1, x._2._2)
          }.sum
          (indx, neighborAverageDensity / lrd)
      }
      .sortBy(_._2, false)
      .cache()

    finalCandiateScores
  }

  /**
    * Select the outlier candidates.
    *
    * @param lofRdd RDD of the entire dataset with lof scores. Each element in the form of
    *               Tuple(Index, (vector, kDistance, density, lof))
    * @return An RDD of outlier candidates. Each element in the form of
    *         Tuple(Index, (vector, kDistance, density, lof))
    * */
  private def getOutlierCandiates(
    lofRdd: RDD[(Long, (Vector, Double, Double, Double))]
  ) = {
    val dataSize = lofRdd.count()
    val candidateRatio = math.min($(contamination) + $(contaminExtend), 1.0)
    val candidateSizeL = (dataSize * candidateRatio).toLong

    val sc = lofRdd.sparkContext
    val thresholds = sc.collectionAccumulator[Double]

    val localCandidates = lofRdd
      .mapPartitions { points =>
        try {
          val pointArray = points.toArray
          if (pointArray.size.toLong < candidateSizeL) {
            // if so then don't put the local threshold into the thresholds accumulator
            pointArray.toIterator
          } else {
            val candidateSizeI = candidateSizeL.toInt
            val queue = collection.mutable
              .PriorityQueue[(Long, (Vector, Double, Double, Double))](
                pointArray.take(candidateSizeI): _*
              )(Ordering.by(x => -(x._2._4)))
            for (i <- candidateSizeI until pointArray.size) {
              queue += pointArray(i)
              queue.dequeue()
            }
            val thresh = queue.dequeue()
            thresholds.add(thresh._2._4)
            val queBuffer = queue.toBuffer
            queBuffer += thresh
            queBuffer.toIterator
          }
        } catch {
          case t: Throwable =>
            LSHLOF.logger.error(
              "Erorr in computing local candidates. " + t.getMessage,
              t
            )
            Iterator.empty
        }
      }
      .cache()

    // to enforce the transformation of localCandidate for thresholds
    localCandidates.count()

    val thresholdList = thresholds.value.asScala
    val middleRdd =
      if (!thresholdList.isEmpty)
        // it means all the partitions have a number of points less than candidate size
        localCandidates.filter(_._2._4 >= thresholdList.max)
      else
        localCandidates

    middleRdd
      .sortBy(_._2._4, false)
      .zipWithIndex()
      .filter(_._2 < candidateSizeL)
      .map { case (x, _) => x }
  }

  /**
    * The distributed computation of LOF based on each individual partition.
    * Locality sensitive-hashing is used for data partitioning.
    *
    * @param data An RDD of the input indexed vectors
    * @return An RDD of the entire dataset with lof scores. Each element in the form of
    *         Tuple(Index, (vector, kDistance, density, lof))
    * */
  //(ID, (vect, kDistance, density, lof))
  private def distributedLOF(
    data: RDD[(Long, Vector)]
  ): RDD[(Long, (Vector, Double, Double, Double))] = {
    val sc = data.sparkContext

    val dim = data.first()._2.size
    val lsh =
      TwoLayerLSH.createTwoLayeredLSH($(numFunctions), dim, $(w), $(seed))
    val bcLSH = sc.broadcast(lsh)

    val lshRdd = data
      .map {
        case (indx, vect) =>
          val hashValue = bcLSH.value.hashFunc(vect)
          (indx, (vect, hashValue))
      }
      .sortBy(
        { case (_, (_, hashValue)) => hashValue },
        ascending = true,
        numPartitions = $ { numPartitions }
      )

    val lofRdd =
      lshRdd.mapPartitions(
        { points =>
          try {
            val pointArray = points.toArray
            val indexedVectors = pointArray.map {
              case (idx, (vect, hashValue)) => (idx, vect)
            }
            val lofArray = LOFPerPartition(indexedVectors)
            assert(pointArray.size == lofArray.size)
            lofArray.iterator
          } catch {
            case t: Throwable => {
              LSHLOF.logger
                .error("Erorr in computing lofRdd. " + t.getMessage, t)
              Iterator.empty
            }
          }
        },
        true
      )

    lofRdd
  }

  /**
    * Compute the LOF score for based on each partition.
    *
    * @param points an array of indexed vectors
    * @return the computed lof scores and other related information for each data point, in the form of
    *         Array(index, (vecotr, kDistance, density, lof))
    * */
  private def LOFPerPartition(
    points: Array[(Long, Vector)]
  ): Array[(Long, (Vector, Double, Double, Double))] = {

    val pointsArray = points.map(_._2)
    val actualNumNeighbors = math.min($(numNeighbors), pointsArray.size - 1)

    val neighborArray = computeNeighborArray(pointsArray, actualNumNeighbors)

    // density
    val densityArray = new Array[Double](pointsArray.size)
    for (ind1 <- 0 until pointsArray.size) {
      var distSum: Double = 0.0
      val neighborhood: Array[(Int, Double)] = neighborArray(ind1)
      if (neighborhood.size == 0) {
        throw new Exception(
          s"neighborhood size == 0!\n knn numNeighbors =${actualNumNeighbors}\n " +
            s"partition record size = ${points.size}\n"
        )
      }

      for (ind2 <- 0 until neighborhood.size) {
        val (neighborIndx, neighborDist) = neighborhood(ind2)
        val neighborKDist: Double =
          (neighborArray(neighborIndx)(neighborArray(neighborIndx).size - 1))._2
        distSum += math.max(neighborDist, neighborKDist)
      }

      densityArray(ind1) = neighborhood.size / distSum
    }

    //lof
    val lofArray = new Array[Double](pointsArray.size)
    for (ind1 <- 0 until pointsArray.size) {
      val neighborhood: Array[(Int, Double)] = neighborArray(ind1)
      val lrdSum: Double = neighborhood
        .map {
          case (neighborIndx, _) =>
            densityArray(neighborIndx)
        }
        .reduce(_ + _)
      lofArray(ind1) = lrdSum / densityArray(ind1) / neighborhood.size
    }

    //results
    points.zipWithIndex.map {
      case ((longIndx, vector), indx) =>
        val kDistance = (neighborArray(indx)(neighborArray(indx).size - 1))._2
        (longIndx, (vector, kDistance, densityArray(indx), lofArray(indx)))
    }

  }

  /**
    * Perform k-nn search for each data instance in the array.
    *
    * @param pointsArray input data, an array of vectors
    * @param numNeighbors the number of neighbors, as k in k-NN
    * @return a 2D array containing a list of nearest neighbors for each data point.
    *         Tuple(index, distance). The index is array-wise.
    * */
  private def computeNeighborArray(
    pointsArray: Array[Vector],
    numNeighbors: Int
  ): Array[Array[(Int, Double)]] = {
    val neighborArray = new Array[Array[(Int, Double)]](pointsArray.size)

    for (ind1 <- 0 until pointsArray.size) {
      val neighborQueue = new KNNPriorityQueue[Int](numNeighbors)
      for (ind2 <- 0 until pointsArray.size) {
        if (ind1 != ind2) {
          val tempDistance =
            Vectors.sqdist(pointsArray(ind1), pointsArray(ind2))
          neighborQueue.offer((ind2, tempDistance))
        }
      }
      neighborArray(ind1) = neighborQueue.dequeueALl()
    }
    neighborArray
  }

  /**
    * Select the k-nearest neighbors from an array of data points (pointsArray) for a data point (vect).
    *
    * @param pointIndx the RDD-wise index for the input vector
    * @param vect the target data point
    * @param pointsArray an array of data points to select k-NN from.
    *                    Each element contains Tuple(index, (vector, kDistance, density, lof))
    *
    * @return the selected k-nearest neighbors of the input vector,
    *         as well as the neighbors related information:
    *         Tuple(index, (distance, kdistance, density))
    * */
  private def computeNeighborhood(
    pointIndx: Long,
    vect: Vector,
    pointsArray: Array[(Long, (Vector, Double, Double, Double))],
    numNeighbors: Int
  ): Array[(Long, (Double, Double, Double))] = {

    val neighborhoodQueue = new KNNPriorityQueue[Int](numNeighbors)

    //using local index in KNNPriorityQueue
    for (localIndx <- 0 until pointsArray.size) {
      val guestTuple = pointsArray(localIndx)
      val guestIndx = guestTuple._1
      val guestVector = guestTuple._2._1
      if (pointIndx != guestIndx) {
        val tempDistance = Vectors.sqdist(vect, guestVector)
        neighborhoodQueue.offer((localIndx, tempDistance))
      }
    }

    neighborhoodQueue.dequeueALl().map {
      case (localIndx, distance) =>
        val guestTuple = pointsArray(localIndx)
        (guestTuple._1, (distance, guestTuple._2._2, guestTuple._2._3))
    }
  }

  /**
    * Merge two neighborhoods.
    * @param neighborhood1 the first neighborhood: Tuple(index, (distance, kdistance, density))
    * @param neighborhood2 the second neighborhood
    * @return the output merged neighborhood
    * */
  private def mergeNeigbhorhood[T](
    neighborhood1: Array[(T, (Double, Double, Double))],
    neighborhood2: Array[(T, (Double, Double, Double))],
    numNeighbors: Int
  ): Array[(T, (Double, Double, Double))] = {
    if (neighborhood1.size == 0)
      neighborhood2
    else if (neighborhood2.size == 0)
      neighborhood1
    else {
      var zeroExists = false
      val buffer = new ArrayBuffer[(T, (Double, Double, Double))](
        neighborhood1.size + neighborhood2.size
      )
      var kCount = 0
      var q1, q2 = 0
      while (q1 < neighborhood1.size && q2 < neighborhood2.size && kCount < numNeighbors) {
        if (neighborhood1(q1)._2._1 <= neighborhood2(q2)._2._1) {
          buffer += neighborhood1(q1)
          q1 += 1
          if (neighborhood1(q1 - 1)._2.equals(0d)) {
            if (!zeroExists) {
              zeroExists = true
              kCount += 1

            }
          } else {
            kCount += 1
          }
        } else {
          buffer += neighborhood2(q2)
          q2 += 1
          if (neighborhood2(q2 - 1)._2.equals(0d)) {
            if (!zeroExists) {
              zeroExists = true
              kCount += 1
            }
          } else {
            kCount += 1
          }
        }
      }

      if (kCount < numNeighbors) {
        while (q1 < neighborhood1.size && kCount < numNeighbors) {
          buffer += neighborhood1(q1)
          q1 += 1
          if (neighborhood1(q1 - 1)._2.equals(0d)) {
            if (!zeroExists) {
              zeroExists = true
              kCount += 1

            }
          } else {
            kCount += 1

          }
        }
        while (q2 < neighborhood2.size && kCount < numNeighbors) {
          buffer += neighborhood2(q2)
          q2 += 1
          if (neighborhood2(q2 - 1)._2.equals(0d)) {
            if (!zeroExists) {
              zeroExists = true
              kCount += 1

            }
          } else {
            kCount += 1

          }
        }
      }
      buffer.toArray
    }
  }

  /**
    * Convert a Dataset into RDD[Vector]
    * */
  def columnToVector(dataset: Dataset[_], colName: String): RDD[Vector] = {
    dataset.select(DatasetUtils.columnToVector(dataset, colName)).rdd.map {
      case Row(point: Vector) => point
    }
  }

}

object LSHLOF {

  @transient lazy val logger = LogManager.getLogger(LSHLOF.getClass)

  /** Column name*/
  private[outlier] val index = "index"

  /** Column name*/
  private[outlier] val lof = "lof"
}
