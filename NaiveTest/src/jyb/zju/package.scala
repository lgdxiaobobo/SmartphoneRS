package jyb

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import scala.math._

import java.text.SimpleDateFormat

package object zju {
  // define the sample class
  case class Sample(u: Int, d: Int, xud: Int)
  // define spark configuration
  def defineSpark(name: String): SparkContext = {
    val sc = new SparkContext(new SparkConf().setAppName(name))
    sc.setLogLevel("WARN")
    sc
  }
  // get datetime based on seconds
  def getDate(sec: Int, format: String): String = {
    new SimpleDateFormat(format).format(sec * 1000L)
  }
  // my add operation
  def add(a: Int, b: Int): Int = a + b
  def add(a: Double, b: Double): Double = a + b
  // my pow2 opeartion
  def pow2(a: Double): Double = a * a
  // load train-set and formated
  def loadTrainSet(sc: SparkContext): RDD[Sample] = {
    val trainFileName = "/pnrs/source/train_parts"
    val merged = sc.textFile(trainFileName).map(_.split('|'))
      .map(ps => (ps(1).toInt, ps(2).toInt))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    println("[INFO] train-sample distribution")
    merged.map(_._2.size).map(x => (x, 1)).reduceByKey(add)
      .sortBy(-_._2).map(p => "%d,%d".format(p._1, p._2)).foreach(println)
    val data = merged.map{case (u, ds) => (u, ds.toArray)}
      .flatMap{case (u, ds) => ds.map(d => Sample(u, d, 1))}
    data.repartition(64)
  }
  // strained split by labels (e.g., sampling with same ratio about positive and negative dataset)
  def strainedSplitWithNeg(data: RDD[Sample], rng: MyRNG, alpha: Double): (RDD[Sample], RDD[Sample]) = {
    val onlyPos = data.filter(_.xud == 1)
    val onlyNeg = data.filter(_.xud == 0)
    val (n1, r1) = rng.getInt
    val seed1 = n1 + onlyPos.count()
    val (pos1, pos2) = strainedSplit(onlyPos, MyRNG(seed1), alpha)
    val (n2, _) = r1.getInt
    val seed2 = n2 + onlyNeg.count()
    val (neg1, neg2) = strainedSplit(onlyNeg, MyRNG(seed2), alpha)
    (pos1.union(neg1), pos2.union(neg2))
  }
  // stained split by labels
  def strainedSplit(data: RDD[Sample], rng: MyRNG, alpha: Double): (RDD[Sample], RDD[Sample]) = {
    val N = data.count()
    val merged = data.map(p => (p.u, Array(p)))
      .reduceByKey(_ ++ _)
    val part0 = merged.filter(_._2.length == 1)
      .map(p => p._2.head)
    val N0 = part0.count()
    val N1 = N - N0
    val beta = min(1.0, N * (1.0 - alpha) / N1)
    // println("[INFO] %f for beta" format beta)
    val part1 = merged.filter(_._2.length > 1)
      .map(_._2).map(randSplit(rng, beta))
    val part10 = part0.union(part1.flatMap(_._1))
    val part11 = part1.flatMap(_._2)
    (part10.cache(), part11.cache())
  }
  // random indexed a list
  def getIndexLst(num: Int, r0: MyRNG): (Array[Int], MyRNG) = {
    if (num == 0) (Array[Int](), r0)
    else{
      val (n1, r1) = r0.getInt
      val (lst, r2) = getIndexLst(num-1, r1)
      (lst :+ n1, r2)
    }
  }
  // random split (ignoring labels)
  def randSplit(rng: MyRNG, beta: Double)(ps: Array[Sample]): (Array[Sample], Array[Sample]) = {
    val (i0, _) = rng.getInt
    val r1 = MyRNG(i0 * ps(0).u)
    val temp = ps.foldLeft((Array[Sample](), Array[Sample](), r1))(
      (agg, x) => {
        val train = agg._1
        val valid = agg._2
        val rand0 = agg._3
        val (prop, rand1) = rand0.getDouble
        if (prop < beta) (train, valid :+ x, rand1)
        else (train :+ x, valid, rand1)
      }
    )
    (temp._1, temp._2)
  }
  // brief static about dataset
  def staticDataset(data: RDD[Sample], withNeg: Boolean): Boolean = {
    val size = data.count()
    val users = data.map(_.u).distinct()
    val items = data.map(_.d).distinct()
    val uNum = users.count()
    val iNum = items.count()
    println("[INFO] static on dataset with #records,#users,#items")
    println("%d,%d,%d".format(size, uNum, iNum))
    if (withNeg){
      val positive = data.filter(_.xud == 1).count()
      val negative = data.filter(_.xud == 0).count()
      println("[INFO] positive v.s. negative as %d:%d".format(positive, negative))
    }
    true
  }
  // brief static about dataset
  def staticDataset(data: RDD[Sample]): Boolean = {
    val size = data.count()
    val users = data.map(_.u).distinct()
    val items = data.map(_.d).distinct()
    val uNum = users.count()
    val iNum = items.count()
    println("[INFO] static on dataset with #records,#users,#items")
    println("%d,%d,%d".format(size, uNum, iNum))
    true
  }
  // generate a fixed size array randomly
  def generate(d: Int, r0: MyRNG): (Array[Double], MyRNG) = {
    if (d == 0) (Array[Double](), r0)
    else{
      val (x, r1) = r0.getDouble
      val (lst, r2) = generate(d-1, r1)
      (x +: lst, r2)
    }
  }
  // add negative feedback manually
  def addNegative(used: Set[Int], items: Array[Int], times: Int, rng: MyRNG):
  Array[(Int, Int)] = {
    val positive = used.toArray.map(d => (d, 1))
    val negative = items.filter(d => !used.contains(d))
      .map(d => (d, 0))
    val N = negative.length
    val M = times * positive.length
    times match {
      case 0 => positive
      case -1 => positive ++ negative
      case _ => {
        val zero = Array[(Int, Int)]()
        val sampler = negative.foldLeft((zero, N, rng))(
          (agg, x) => {
            val m0 = agg._1.length
            val n0 = agg._2
            val r0 = agg._3
            val (prop, r1) = r0.getDouble
            val ratio = (M - m0) * 1.0 / n0
            if (prop < ratio) (agg._1:+x, n0-1, r1)
            else (agg._1, n0-1, r1)
          }
        )._1
        positive ++ sampler
      }
    }
  }

}
