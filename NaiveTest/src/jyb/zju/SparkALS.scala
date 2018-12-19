package jyb.zju

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

import math.{pow, sqrt}
/*
  build recommendation system through MLLIB ALS
*/
object SparkALS {

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("mllib ALS")

    // from stack-overflow solution
    sc.setCheckpointDir("/pnrs/MyCheckPoints")

    val train = loadTrainSet(sc).map(p => (p.u, p.d))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)

    train.take(1).foreach(println)

    val items = train.flatMap(_._2).distinct().collect()
    println(items.length)
    val itemBD = sc.broadcast(items)

    val rank = 20
    // val lu = 0.01
    // val ld = 0.01
    val lambda = 0.01
    val numIteration = 30

    for (cnt <- 1 to 10){
      val rng = MyRNG(42 + cnt)

      val addedData = train.mapValues(p => addNegative(p, itemBD.value, 1, rng))
        .flatMap{case (user, usage) => usage.map(p => Sample(user, p._1, p._2))}
      staticDataset(addedData, true)
      val (part1, part2) = strainedSplitWithNeg(addedData, rng, 0.8)
      staticDataset(part1, true)
      staticDataset(part2, true)

      val ratings = part1.map(p => Rating(p.u, p.d, p.xud)).cache()
      val model = ALS.train(ratings, rank, numIteration, lambda)
      println("Finished ALS Training")

      val usersProducts = ratings.map { case Rating(user, product, _) => (user, product) }
//      println(usersProducts.count())

      val predictions = model.predict(usersProducts).map{
        case Rating(user, product, rate) => ((user, product), rate)
      }
//      println(predictions.count())

      val ratesAndPreds = ratings.map { case Rating(user, product, rate) => ((user, product), rate) }
        .join(predictions)
//      println(ratesAndPreds.count())

      val MSE = ratesAndPreds.map { case (_, (r1, r2)) => pow(r1 - r2, 2.0) }.mean()
      println("Mean Squared Error = " + MSE)

      val arrange1 = part1.filter(_.xud == 1).map(p => (p.u, p.d)).aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val arrange2 = part2.filter(_.xud == 1).map(p => (p.u, p.d)).aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val changed = arrange1.join(arrange2).mapValues(p => p._2.diff(p._1))
        .filter(_._2.nonEmpty)
      println("%d users changed devices".format(changed.count()))

      val recommendation = model.recommendProductsForUsers(15)
        .mapValues(rates => rates.map{case Rating(_, d, rud) => (d, rud)})
        .join(arrange1).mapValues{case (products, used) => products.filter(p => !used.contains(p._1))}
        .mapValues(products => products.take(10)) //.map(_._1).take(10))
      recommendation.take(2).foreach{
        case (u, ps) =>
          println(u)
          println(ps.map(p => "%d,%f".format(p._1, p._2)).mkString("|"))
      }
      val merged = changed.join(recommendation.mapValues(_.map(_._1)))
        .mapValues(p => (p._1, p._2)).map(_._2)
      println(merged.count())
      merged.take(2).foreach{
        case (t, p) =>
          println(t.mkString("|"))
          println(p.mkString("|"))
      }
      for (k <- 1 to 10){
        val mergedK = merged.map{case (t, p) => (t, t.intersect(p.take(k).toSet))}
        val precision = mergedK.map(c => c._2.size * 1.0 / k).mean()
        val recall = mergedK.map(c => c._2.size * 1.0 / c._1.size).mean()
        println("%d,%f,%f".format(k, precision, recall))
      }
    }
  }
}
