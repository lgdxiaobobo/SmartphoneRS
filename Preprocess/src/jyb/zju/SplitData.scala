package jyb.zju

import java.io.{File, FileWriter}

// 按照时间段划分数据
// 由于模型目标根据过去的使用行为对未来使用进行推荐
// 所以这里按照时间段来对样本进行划分
// 训练集和测试集 20161023-20170124？ 20170201-20170215
// 先匹配智能设备名称，然后按照名称分组

object SplitData {

  def isTrain(date: String): Boolean = {
    date >= "20161023" && date <= "20170124"
  }

  def isTest(date: String): Boolean = {
    date >= "20170201" && date <= "20170215"
  }

  def main(args: Array[String]): Unit = {

    val sc = defineSpark("split dataset by dates")
    // 对于20161023-20170124作为训练集
    //    20170201-20170215作为测试集
    val data = sc.textFile("/pnrs/source/numbered_dataset")
      .map(_.split('|'))
    val dates = data.map(_.head).distinct()
    println("[INFO] totally %d dates".format(dates.count()))
    dates.collect().sorted.foreach(println)

    // 分别统计训练集和测试集的信息
    val train = data.filter(ps => isTrain(ps(0)))
    val test = data.filter(ps => isTest(ps(0)))
    val users1 = train.map(_.apply(1)).distinct()
    val users2 = test.map(_.apply(1)).distinct()
    val items1 = train.map(_.last).distinct()
    val items2 = test.map(_.last).distinct()
    // 基本信息
    val t1Num = train.count()
    val t2Num = test.count()
    val u1Num = users1.count()
    val u2Num = users2.count()
    val i1Num = items1.count()
    val i2Num = items2.count()
    println("[INFO] in train-set %d records, %d users and %d items".format(t1Num, u1Num, i1Num))
    println("[INFO] in test-set %d records, %d users and %d items".format(t2Num, u2Num, i2Num))
    // 关于能否后续验证以及分层抽样的统计
    // 筛选到测试集的是否一定是换机用户？
    val trainMerge = train.map(ps => (ps(1), ps(2)))
      .aggregateByKey(Set[String]())(_ + _, _ ++ _)
    val testMerge = test.map(ps => (ps(1), ps(2)))
      .aggregateByKey(Set[String]())(_ + _, _ ++ _)
    val common = trainMerge.join(testMerge)
      .map{case (_, (old, now)) => now.diff(old)}
      .filter(_.nonEmpty)
    println("[INFO] %d users change devices comparing with train and test parts" format common.count())
    // 分层抽样的话主要根据训练集用户使用手机分布来确定
    val userUsedCont = trainMerge.map(_._2).map(_.size).map(x => (x, 1))
      .reduceByKey(add).sortBy(_._1)
    userUsedCont.collect().foreach{case (cont, time) => println("%d,%d".format(cont, time))}
    // 将训练集和测试集先保存到本地吧，考虑用python写一个local的算法
    val f1 = new FileWriter(new File("./train_parts"))
    train.map(_.mkString("|")).map(_ + "\n")
      .collect().foreach(f1.write)
    f1.close()
    val f2 = new FileWriter(new File("./test_parts"))
    test.map(_.mkString("|")).map(_ + "\n")
      .collect().foreach(f2.write)
    f2.close()
  }

}
