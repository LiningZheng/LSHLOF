package org.apache.spark.ml.outlier.utils

import scala.util.Random
import breeze.linalg.normalize
import org.apache.spark.ml.linalg._

/**
  * A two-layered locality sensitive-hashing structure.
  *
  * @param numFunctions number of hash functions for in first layer
  * @param dim dimensionality of the random unit vectors in the first layer
  * @param w bucket length (width) of the hash functions in the first layer
  * @param seed seed used in generating random numbers.
  * */
private[ml] class TwoLayerLSH private (private var numFunctions: Int,
                                       private var dim: Int,
                                       private var w: Double,
                                       private var seed: Long)
    extends Serializable {

  private var _randUnitVectors = None: Option[Array[Vector]]
  private var _secondLayerVector = None: Option[Vector]
  private var _bArray = None: Option[Array[Double]]

  private def randUnitVectors = _randUnitVectors.get
  private def secondLayerVector = _secondLayerVector.get
  private def bArray = _bArray.get

  private def initialize(): this.type = {
    val rand = new Random(seed)
    _randUnitVectors = Some(generateRandUnitVectorArray)
    _secondLayerVector = Some(generateRandUnitVector(rand, this.numFunctions))
    _bArray = Some(Array.fill(this.numFunctions)(rand.nextDouble()))
    this
  }

  private def generateRandUnitVectorArray() = {
    val rand = new Random(seed)

    Array.fill(this.numFunctions) {
      generateRandUnitVector(rand, this.dim)
    }
  }

  private def generateRandUnitVector(rand: Random, size: Int) = {
    val singleArray = Array.fill(size)(rand.nextGaussian())
    val normalizedBreezeVector = normalize(breeze.linalg.Vector(singleArray))
    Vectors.fromBreeze(normalizedBreezeVector)
  }

  /**
    * Compute the final hash value for an input vector
    * @param vector input vector
    * */
  def hashFunc(vector: Vector): Double = {
    val firstLayerRes = randUnitVectors.zipWithIndex
      .map {
        case (randVector, indx) =>
          Math.floor(BLAS.dot(vector, randVector) / w + bArray(indx))
      }

    BLAS.dot(Vectors.dense(firstLayerRes), secondLayerVector)
  }

}

/**
  * Companion object that provides a creation method
  * */
private[ml] object TwoLayerLSH {
  def createTwoLayeredLSH(numFunctions: Int,
                          dim: Int,
                          w: Double = 4.0,
                          seed: Long) = {
    new TwoLayerLSH(numFunctions, dim, w, seed)
      .initialize()
  }
}
