# LSHLOF (Distributed Local Outlier Factor with Locality-Sensitive Hashing)
<a href="https://dl.acm.org/doi/abs/10.1145/342009.335388"> Local Outlier Factor (LOF)</a> is an important outlier detection technique, based on the relative densities among k-nearest neighbors. LOF has distinctive advantages on skewed datasets with regions of various densities. 

This project is an implementation of a distributed version of LOF, using Locality-Sensitive Hashing for data partitioning. More details on the implementation can be found at my <a href="https://ruor.uottawa.ca/handle/10393/39817"> Master's thesis</a>, Chapter 4.2. 

## Example

*Scala API*
```scala
    val spark = SparkSession
      .builder()
      .master("local") // test in local mode
      .appName("LSHLOFExample")
      .getOrCreate()

    // data downloaded from
    // [[https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)]]
    val filePath = "data/breastw.csv"
    val df = spark.read.option("inferSchema", "true").csv(filePath)

    val originalLabelColName = "_c10"
    val indexer = new StringIndexer()
      .setInputCol(originalLabelColName)
      .setOutputCol("label")

    val featureColRange = (1 until 10)
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
```

## Requirements
The current version of LSHLOF is built on Spark 2.4.3 and Scala 2.11.8.

## Feedback
Please feel free to contact me for feedback and suggestions :)
<a href="mailto:zhenglining5@gmail.com">email</a>
