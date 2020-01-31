import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.outlier._
import org.apache.spark.ml.feature.{
  MinMaxScaler,
  StringIndexer,
  VectorAssembler
}

import org.apache.spark.sql.SparkSession

object LSHLOFExample {
  def main(args: Array[String]): Unit = {
    // data downloaded from
    // [[https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)]]
    val filePath = "data/breastw.csv"

    val spark = SparkSession
      .builder()
      .master("local") // test in local mode
      .appName("LSHLOFExample")
      .getOrCreate()

    val df = spark.read.option("inferSchema", "true").csv(filePath)

    val originalLabelColName = "_c10"
    val indexer = new StringIndexer()
      .setInputCol(originalLabelColName)
      .setOutputCol("label")

    val featureColRange = 1 until 10
    val featureColNames = featureColRange.toArray.map { i =>
      s"_c${i}"
    }
    val assembler = new VectorAssembler()
      .setInputCols(featureColNames)
      .setOutputCol("unscaled_features")

    //scaling has a significant impact on the detection accuracy for certain datasets
    val scaler = new MinMaxScaler()
      .setInputCol("unscaled_features")
      .setOutputCol("features")

    val lofOD = new LSHLOF()
      .setContamination(0.1)
      .setContaminExtend(0.1)
      .setNumPartitions(3)
      .setNumNeighbors(30)
      .setNumFunctions(5)

    val pipeline =
      new Pipeline().setStages(Array(indexer, assembler, scaler, lofOD))
    val model = pipeline.fit(df)
    val results = model.transform(df)
    results.show()
  }
}
