package jyb.zju

import java.text.SimpleDateFormat
import java.io.{File, FileWriter}

object FilterInvalid {

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("filter users with more than 5 devices and static devices distribution")
    val data = sc.textFile("/pnrs/user_with_device_related").map(_.split('|'))
    val sec2Day: Int => String = sec => new SimpleDateFormat("yyyyMMdd").format(sec * 1000L)
    val mergedByUser = data.map(ps => (ps(0), (sec2Day(ps(1).toInt), ps(2))))
      .aggregateByKey(Array[(String, String)]())(_ :+ _, _ ++ _)
    println("[INFO] totally %d users".format(mergedByUser.count()))
    val invalidUsers = mergedByUser.map{case (u, ds) => (u, ds.length)}
      .sortBy(-_._2).filter(_._2 >= 5)
    invalidUsers.take(20).foreach(p => println("%s,%d".format(p._1, p._2)))
    println("[INFO] %d invalid users" format invalidUsers.count())
    val validUsage = mergedByUser.filter{case (_, ds) => ds.length < 5}
    println("[INFO] remains %d users".format(validUsage.count()))
    val deviceCont = validUsage.flatMap(_._2).map(_._2)
      .map(x => (x, 1)).reduceByKey(add).sortBy(-_._2)
    // val contLocal = deviceCont.map{case (name, cont) => "%s,%d\n".format(name, cont)}.collect()
    // val wr = new FileWriter(new File("./valid_device_cont"))
    // contLocal.foreach(wr.write)
    // wr.close()
    val enoughRate = args(0).toInt * 0.01
    println("[INFO] search top-%f devices as items".format(enoughRate))
    // find proper threshold
    val accumulate = deviceCont.collect().foldLeft(Array[Int]())(
      (agg, x) => {
        if (agg.isEmpty) agg :+ x._2
        else agg :+ (agg.last + x._2)
      }
    )
    val totalUsers = accumulate.last
    val enoughDeviceNum = accumulate.indexWhere(x => x > enoughRate * totalUsers)
    println("[INFO] we need select %d devices".format(enoughDeviceNum))
    val devices = deviceCont.take(enoughDeviceNum).map(_._1).zipWithIndex
    val deviceDict = sc.broadcast(devices.toMap)
    val filtered = validUsage.flatMap{case (u, ds) => ds.map(d => (u, d))}
      .filter{case (u, (_, d)) => deviceDict.value.contains(d)}
    val static = filtered.map(p => (p._1, Array(p._2._2))).reduceByKey(_ ++ _)
      .map(p => (p._2.length, 1)).reduceByKey(add).sortBy(-_._2).collect()
    static.foreach(p => println("%d,%d".format(p._1, p._2)))

    val f1 = new FileWriter(new File("./numbered_device"))
    devices.map{case (name, idx) => "%s,%d\n".format(name, idx)}
        .foreach(f1.write)
    f1.close()
    val f2 = new FileWriter(new File("./numbered_users"))
    val u2Index = filtered.map(_._1).distinct().zipWithIndex()
    u2Index.map{case (user, idx) => "%s,%d\n".format(user, idx)}
      .collect().foreach(f2.write)
    f2.close()
    val f3 = new FileWriter(new File("./numbered_dataset"))
    val u2id = sc.broadcast(u2Index.collect().toMap)
    val d2id = sc.broadcast(devices.toMap)
    println("[INFO] remains %d records".format(filtered.count()))
    filtered.map{case (u, (t, d)) => "%s|%d|%d\n".format(t, u2id.value(u), d2id.value(d))}
        .collect().foreach(f3.write)
    f3.close()
  }

}
