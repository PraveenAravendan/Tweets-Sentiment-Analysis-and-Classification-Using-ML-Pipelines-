package main.scala

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object tweets {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage (User parameters order): TweetClassification LocationOfCSV OutputDir")
    }
    // Input directory (File Path), Output directory (File Path)
    val inputFilePath = args(0)
    val outputFilePath = args(1)
    
    //Spark Session
    val sparkConf = new SparkConf().setAppName("Tweet Classification").setMaster("local").set("spark.driver.host", "localhost")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession: SparkSession = SparkSession.builder.appName("Tweet Classification").getOrCreate()
    
    var airlineTweets = sparkSession.read.option("header","true").option("inferSchema","true").csv(inputFilePath)
    airlineTweets = airlineTweets.drop("negativereason","negativereason_confidence","airline","airline_sentiment_gold","name","negativereason_gold","retweet_count","tweet_coord","tweet_created","tweet_location","user_timezone","airline_sentiment_confidence")
    var TweetsFiltered =  airlineTweets.filter(col("text").isNotNull) //remove null values
    
    //Setting up steps for pipeline to execute
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol(stopWordsRemover.getOutputCol).setOutputCol("features").setNumFeatures(1000)
    val indexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.1)
    val pipe = new Pipeline().setStages(Array(tokenizer,stopWordsRemover,hashingTF,indexer,lr))
    
    //Training and Test Data
    val trainAndTestData = TweetsFiltered.randomSplit(Array[Double](0.7, 0.3), 18)
    val train = trainAndTestData(0)
    val test = trainAndTestData(1)
    
    val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr.regParam, Array(0.1, 0.01)).build()
    // Creating a crossValidator to allow us to choose best parameters
    val cross_valid = new CrossValidator().setEstimator(pipe).setEvaluator(new MulticlassClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(8)

    // Running crossValidation so that we can choose the best set of parameters.
    val cvModel = cross_valid.fit(train)
    val predResult = cvModel.bestModel.transform(test)
    val finalResult = predResult.select("prediction", "label").rdd.map {case Row(label: Double, prediction: Double) => (prediction, label)}


    // Create metrics object
    val metrics = new MulticlassMetrics(finalResult)
    
    // Metrics Result
    val metricsResultRDD =  sparkSession.sparkContext.parallelize(Seq(
      ("Confusion matrix", metrics.confusionMatrix),
      ("Accuracy", metrics.accuracy),
      ("Weighted precision", metrics.weightedPrecision),
      ("Weighted recall", metrics.weightedRecall),
      ("Weighted F1 score", metrics.weightedFMeasure),
      ("Weighted false positive rate", metrics.weightedFalsePositiveRate)

    ))
    //Writing to output File
    metricsResultRDD.saveAsTextFile(outputFilePath)
    
    }
}