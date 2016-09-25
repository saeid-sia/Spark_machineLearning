
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegression {

  val conf = new SparkConf().setAppName("ClassificationSpark")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val lines = sc.textFile("/home/data.txt").cache()

    val lable: RDD[(Double, Vector)] = lines.map(line => line.split(","))
      .map(elem => {
      (
        if (elem(0).equals("m")) 1.0 else 0.0,Vectors.dense(elem.slice(1, elem.length).map(arr => arr.toDouble))
        )
    }
      )

    val splitdata = lable.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splitdata(0).cache()
    val test = splitdata(1)

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val df: DataFrame = training.toDF("label", "features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(8)

    val lr = new LogisticRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(pca, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val model: PipelineModel = pipeline.fit(df)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvmodel = cv.fit(df)

    val raw: DataFrame = test.toDF("label", "features")

    val valuep: RDD[Row] = cvmodel.transform(raw).select("label", "prediction").rdd
    val MSE = valuep.map{case Row(label: Double, prediction: Double) =>
      math.pow(label - prediction, 2)}.mean()

    println("training Mean Squared Error = " + MSE)
  }
}
