import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.log4j.Logger
import org.apache.log4j.Level
object Training {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val personalRatingsFile = "C:\\Users\\Вон тот парень\\IdeaProjects\\Pr6-master\\src\\sets\\personalRatings.txt"
  val MovieFile = "C:\\Users\\Вон тот парень\\IdeaProjects\\Pr6-master\\src\\sets\\movies.dat"
  val RatingsFile = "C:\\Users\\Вон тот парень\\IdeaProjects\\Pr6-master\\src\\sets\\ratings.dat"
  val UsersFile = "C:\\Users\\Вон тот парень\\IdeaProjects\\Pr6-master\\src\\sets\\users.dat"
  val movieLensHomeDir = "C:\\Users\\Вон тот парень\\IdeaProjects\\Pr6-master\\src\\sets\\"

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  case class UserMovie(userId: Int, movieId: Int)
  case class Movie(movieId: Int, movieName: String, rating: Float)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    return Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]) {

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate()

    import sparkSession.implicits._

    //Load Ratings
    val ratings = sparkSession
      .read.textFile(movieLensHomeDir + "ratings.dat")
      .map(parseRating)
      .toDF()

    //Load Movies
    val moviesRDD = sparkSession
      .read.textFile(movieLensHomeDir + "movies.dat").map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1))
    }
    //Load Movies
    val predictRDD = sparkSession
      .read.textFile(movieLensHomeDir + "predict.dat").map { line =>
      val fields = line.split("::")
      UserMovie(fields(0).toInt, fields(1).toInt)
    }

    //Load my ratings
    val myRating = sparkSession.read.textFile(movieLensHomeDir + "personalRatings.txt")
      .map(parseRating)
      .toDF()

    //show the DataFrames
    ratings.show(10)
    myRating.show(10)

    val numRatings = ratings.distinct().count()
    val numUsers = ratings.select("userId").distinct().count()
    val numMovies = moviesRDD.count()

    val ratingsWithMy = ratings.union(myRating)

    //get movies dictionary
    val movies = moviesRDD.collect.toMap

    val Array(training, test) = ratingsWithMy.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //Get trained model
    val model = als.fit(training)

    //Evaluate Model Calculate RMSE
    val predictions = model.transform(test).na.drop
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root-mean-square error = $rmse")

    //Get My Predictions
    val predict = predictRDD.toDF()
    val myPredictions = model.transform(predict).na.drop

    //Show your recomendations
    val myMovies = myPredictions.map(r => Movie(r.getInt(1), movies(r.getInt(1)), r.getFloat(2))).toDF
    myMovies.show(100)



  }
}





