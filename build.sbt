name := "LSH"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.3"

libraryDependencies ++= Seq(
  // Last stable release
  "com.typesafe" % "config" % "1.3.2",
  "org.scalanlp" %% "breeze" % "0.13.2",
  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly ()
)
