package jyb

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import scala.math._

import java.text.SimpleDateFormat

package object zju {
  // My Friend class
  // numbered name and similarity score
  case class Friend(name: Int, score: Double){
    def mapScore(func: Double => Double): Friend = Friend(name, func(score))
    override def toString: String = "%d,%f".format(name, score)
  }
  // My latent factor as a vector
  // several essencial operation function for vector
  case class Vector(value: Array[Double]){
    def getRank: Int = value.length
    def getNorm2: Double = value.map(pow2).sum
    def getNorm: Double = sqrt(getNorm2)
    def dot(v2: Vector): Double =
      value.zip(v2.value).map(p => p._1 * p._2).sum
    def mul(w: Double): Vector = Vector(value.map(_ * w))
    def add(v2: Vector): Vector = add(v2, 1.0, 1.0)
    def add(v2: Vector, w2: Double): Vector = add(v2, 1.0, w2)
    def add(v2: Vector, w1: Double, w2: Double): Vector =
      Vector(value.zip(v2.value).map(p => w1 * p._1 + w2 * p._2))
    def normVector: Vector = {
      val norm = getNorm
      mul(1.0 / norm)
    }
    def map(func: Double => Double): Vector = {
      Vector(value.map(func))
    }
    def div(w: Double): Vector =
      if (w == 0) this
      else mul(1.0 / w)
  }
  // My Sample class as a tri-tuple (u, d, xud)
  case class Sample(u: Int, d: Int, xud: Int)

  def defineSpark(name: String): SparkContext = {
    val sc = new SparkContext(new SparkConf().setAppName(name))
    sc.setLogLevel("WARN")
    sc
  }

  def getDate(sec: Int, format: String): String = {
    new SimpleDateFormat(format).format(sec * 1000L)
  }

  def add(a: Int, b: Int): Int = a + b
  def add(a: Double, b: Double): Double = a + b

  def pow2(a: Double): Double = a * a

  def loadTrainSet(sc: SparkContext): RDD[Sample] = {
    val trainFileName = "/pnrs/source/train_parts"
    val merged = sc.textFile(trainFileName).map(_.split('|'))
      .map(ps => (ps(1) toInt, ps(2) toInt))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    println("[INFO] train-sample distribution")
    merged.map(_._2.size).map(x => (x, 1)).reduceByKey(add)
      .sortBy(-_._2).map(p => "%d,%d".format(p._1, p._2)).foreach(println)
    val data = merged.map{case (u, ds) => (u, ds.toArray)}
      .flatMap{case (u, ds) => ds.map(d => Sample(u, d, 1))}
    data.repartition(64)
  }

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

  def getIndexLst(num: Int, r0: MyRNG): (Array[Int], MyRNG) = {
    if (num == 0) (Array[Int](), r0)
    else{
      val (n1, r1) = r0.getInt
      val (lst, r2) = getIndexLst(num-1, r1)
      (lst :+ n1, r2)
    }
  }

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

  def generate(d: Int, r0: MyRNG): (Array[Double], MyRNG) = {
    if (d == 0) (Array[Double](), r0)
    else{
      val (x, r1) = r0.getDouble
      val (lst, r2) = generate(d-1, r1)
      (x +: lst, r2)
    }
  }
  // obtain normalized vector randomly
  def randomVector(d: Int, r0: MyRNG): (Vector, MyRNG) = {
    val (value, r1) = generate(d, r0)
    (Vector(value).normVector, r1)
  }
  // initialize factor for every user
  def factorInit(keys: Array[Int], d: Int, rng: MyRNG): (Map[Int, Vector], MyRNG) = {
    keys.foldLeft((Map[Int, Vector](), rng))(
      (agg, k) => {
        val dct = agg._1
        val r0 = agg._2
        val (tmp, r1) = randomVector(d, r0)
        (dct + (k -> tmp), r1)
      }
    )
  }
  // My log-loss metric function
  val logLoss: (Double, Double) => Double =
    (t, p) => -t*log(p)+(t-1.0)*log(1.0-p)
  // get total loss
  def getLoss(data: RDD[Sample],  p: Map[Int, Vector], q: Map[Int, Vector], func: (Double, Double) => Double): Double = {
    val sc = data.context
    val pBD = sc.broadcast(p)
    val qBD = sc.broadcast(q)
    getLoss(data, pBD, qBD, func)
  }
  // get total loss
  def getLoss(data: RDD[Sample],  p: Map[Int, Vector], q: Map[Int, Vector],
              wd: Broadcast[Map[Int, Double]], func: (Double, Double) => Double): Double = {
    val sc = data.context
    val pBD = sc.broadcast(p)
    val qBD = sc.broadcast(q)
    getLoss(data, pBD, qBD, wd, func)
  }

  def getLoss(data: RDD[Sample],  pBD: Broadcast[Map[Int, Vector]],
              qBD: Broadcast[Map[Int, Vector]], func: (Double, Double) => Double): Double = {
    val remains = data.filter(x => pBD.value.contains(x.u) && qBD.value.contains(x.d))
    val loss = remains.map(x =>
      (x.xud, predict(x.u, x.d, pBD.value, qBD.value, logistic)))
    loss.map(p => func(p._1, p._2)).mean()
  }
  // get rank loss (for rand metric)
  def getRankLoss(data: RDD[MyRank],  p: Map[Int, Vector], q: Map[Int, Vector]): Double = {
    val sc = data.context
    val pBD = sc.broadcast(p)
    val qBD = sc.broadcast(q)
    getRankLoss(data, pBD, qBD)
  }
  // rank loss
  def getRankLoss(data: RDD[MyRank],  pBD: Broadcast[Map[Int, Vector]],
                  qBD: Broadcast[Map[Int, Vector]]): Double = {
    val remains = data.filter(x => pBD.value.contains(x.user) && qBD.value.contains(x.used) && qBD.value.contains(x.unused))
    val loss = remains.map(x => {
      val pu = pBD.value(x.user)
      val qd1= qBD.value(x.used)
      val qd2= qBD.value(x.unused)
      val sigma = pu.dot(qd1) - pu.dot(qd2)
      -log(logistic(sigma))
    })
    loss.mean()
  }

  def getLoss(data: RDD[Sample],  pBD: Broadcast[Map[Int, Vector]],
              qBD: Broadcast[Map[Int, Vector]], wBD: Broadcast[Map[Int, Double]],
              func: (Double, Double) => Double): Double = {
    val remains = data.filter(x => pBD.value.contains(x.u) && qBD.value.contains(x.d))
    val loss = remains.map(x => {
      val wud = if (x.xud == 1) 1.0 else wBD.value(x.d)
      (x.xud, predict(x.u, x.d, pBD.value, qBD.value, logistic), wud)
    })
    loss.map{case (xud, yud, wud) => wud * func(xud, yud)}.mean()
  }
  // My logistic function
  def logistic(x: Double): Double = 1.0 / (1.0 + exp(-x))
  // Predict corresponding preference for user u on item d
  def predict(u: Int, d: Int, p: Map[Int, Vector], q: Map[Int, Vector], func: Double => Double): Double = {
    val pu = p(u)
    val qd = q(d)
    func(pu.dot(qd))
  }

  def getError(data: RDD[Sample], p: Broadcast[Map[Int, Vector]],
               q: Broadcast[Map[Int, Vector]], func: Double => Double):
  RDD[(Sample, Double)] = {
    data.map(x => {
      (x, predict(x.u, x.d, p.value, q.value, func))
    })
  }

  def getRankError(data: RDD[MyRank], p: Broadcast[Map[Int, Vector]],
                   q: Broadcast[Map[Int, Vector]], func: Double => Double):
  RDD[(MyRank, Double)] = {
    data.map(x => {
      val pu = p.value(x.user)
      val qd1= q.value(x.used)
      val qd2= q.value(x.unused)
      val sigma = pu.dot(qd1) - pu.dot(qd2)
      (x, func(sigma))
    })
  }
  // My sqrt-score for popularity
  def sqrtScore(c: Int): Double = 1.0 / (1.0 + sqrt(c))
  // My log-score for popularity
  def logScore(c: Int): Double = 1.0 / (1.0 + log(c))
  // recommend N items with highest scores
  def recTopN(n: Int, P: Map[Int, Vector], Q: Map[Int, Vector])(x: (Int, Set[Int])): (Int, Array[Int]) = {
    val pu = P(x._1)
    val used = x._2
    val topN = Q.toArray.map{
      case (d, qd) =>
        val yud = logistic(pu.dot(qd))
        (d, yud)
    }.sortBy(-_._2).map(_._1).filter(d => !used.contains(d))
    (x._1, topN)
  }
  // add negative samples by times v.s. the size of positive feedback
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

  // def addNegative(used: Set[Int], items: Array[Int], times: Int, rng: MyRNG):
  // Array[(Int, Int)] = {
  //   val positive = used.toArray.map(d => (d, 1))
  //   val negative = items.filter(d => !used.contains(d))
  //     .map(d => (d, 0))
  //   val M = times * positive.length
  //   val ratio = M * 1.0 / negative.size
  //   times match {
  //     case 0 => positive
  //     case -1 => positive ++ negative
  //     case _ => {
  //       val sampler = negative.foldLeft((Array[(Int, Int)](), rng))(
  //         (agg, x) => {
  //           val (prop, r1) = agg._2.getDouble
  //           if (prop < ratio) (agg._1 :+ x, r1)
  //           else (agg._1, r1)
  //         })._1.take(M)
  //       positive ++ sampler
  //     }
  //   }
  // }
  // get ranked sample
  // assume preference ordered
  // used > not used
  def getRankSample(data: RDD[Sample]): RDD[MyRank] = {
    val m1 = data.map(p => (p.u, p))
      .aggregateByKey((Array[Int](), Array[Int]()))(
        (agg, x) => if (x.xud == 0) (agg._1 :+ x.d, agg._2) else (agg._1, agg._2 :+ x.d),
        (a, b) => (a._1 ++ b._1, a._2 ++ b._2)
      )
    m1.flatMap{
      case (u, (zeros, ones)) =>
        ones.flatMap(j => zeros.map(k => MyRank(u, j, k)))
    }
  }
  // normalized the similarities of friends with soft-max
  def normalized(fs: Array[Friend]): Array[Friend] = {
    val softMax = fs.map(_.mapScore(x => exp(-x)))
    val softSum = softMax.map(_.score).sum
    softMax.map(_.mapScore(x => x / softSum))
  }
  // load graph
  def loadNumberedGraph(sc: SparkContext, gname: String): RDD[(Int, Array[Friend])] = {
    val graph = sc.textFile(gname).map(_.split('|'))
    val numberedGraph = graph.map(ps => (ps(0).toInt, Friend(ps(1).toInt, ps(2).toDouble)))
        .aggregateByKey(Array[Friend]())(_ :+ _, _ ++ _).mapValues(normalized)
    numberedGraph
  }

}
