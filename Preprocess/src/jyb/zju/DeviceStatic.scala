package jyb.zju

/*
    Since our dataset contains no more than 3 months,
    users commonly utilize a little phones.
    Thus we suppose to analysis their usage by static
*/
import java.io.{File, FileWriter}
import java.text.SimpleDateFormat

case class Context(btime: Int, device: String){
  def isEquals(other: Context): Boolean = {
    device.equals(other.device)
  }
  def getDay: String = {
    val parser = new SimpleDateFormat("yyyyMMdd")
    parser.format(btime * 1000L)
  }
  override def toString: String = "%d|%s".format(btime, device)
}

object DeviceStatic {

  def formatData(ps: Array[String], dict: Map[String, String]): (String, Context) = {
    val btime = ps(0).toInt
    val msisdn = ps(1)
    // val Some(device) = dict.get(ps(2))
    val device = dict(ps(2))
    (msisdn, Context(btime, device))
  }
  // arrange records by corresponding daytime
  // e.g., arrange records in the same day (there are totally 79 distinct days)
  def distinctUsage(lst: Array[Context]): Array[Context] = {
    val sorted = lst.sortBy(_.btime)
    // 1st formated, d1->d2->d1 => d1->d1
    val fmt1 = sorted.foldLeft(Array[Context]())(
      (agg, x) =>
        if (agg.length > 1){
          val last1 = agg.last
          val drop1 = agg.dropRight(1)
          val last2 = drop1.last
          if (last2.isEquals(x) && !last1.isEquals(x)) drop1 :+ x
          else agg :+ x
        }else agg :+ x
    )
    val fmt2 = fmt1.foldLeft(Array[Context]())(
      (agg, x) =>
        if (agg.isEmpty) agg :+ x
        else{
          if (agg.last.isEquals(x)) agg
          else agg :+ x
        }
    )
    fmt2
  }
  /*
    As users change their phones,
    they would use them for a period.
    In one day, we need not to record all phones.
    Thus in our dataset, 
    we choose the maximum device in one day alternatively.
  */
  def distinctEveryday(lst: Array[Context]): Context = {
    val deviceCont = lst.foldLeft(Map[String, Int]())(
      (agg, x) => {
        val k = x.device
        val v = agg.getOrElse(k, 0) + 1
        agg + (k -> v)
      }
    ).toArray.sortBy(-_._2)
    val popularDevice = deviceCont.head._1
    val popularBtime = lst.filter(p => p.device.equals(popularDevice)).map(_.btime).min
    Context(popularBtime, popularDevice)
  }

  def getDistinctDevice(p: (String, Array[Context])): Seq[String] = {
    p._2.map(_.device).distinct.toSeq
  }

  def concat(str1: String, str2: String): String = "%s#%s".format(str1, str2)
  def divide(str: String): Array[String] = str.split('#')

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("static dataset based on device")
    val needExtract = args(0).toInt
    println(needExtract)
    if (needExtract == 1) {
      val data = sc.textFile("/pnrs/validData/last_idx_*").map(_.split('|'))
      val deviceDict = loadDeviceDict(sc)
      val userUsedDevice = data.map(ps => formatData(ps, deviceDict.value))
      /*
        arrange strategy:
        1. arrange by users
        2. for one user, arrange by days, assume one phone one day
      */
      val arrangeByDate = userUsedDevice.map{
        case (user, usage) => (concat(user, usage.getDay), usage)
      }.aggregateByKey(Array[Context]())(_ :+ _, _ ++ _)
      arrangeByDate.map(_._1).take(10).foreach(println)
      // rid-off phones with the MFU one daily
      // learn users' daily usage about phone
      val usedEverydayUniq = arrangeByDate.map(p => (divide(p._1), distinctEveryday(p._2)))
        .map{case (ps, usage) => (ps(0), usage)}
      // arrange phone usage behaviors by users
      val mergeByUser = usedEverydayUniq.aggregateByKey(Array[Context]())(_ :+ _, _ ++ _)
      println("[INFO] %d users here".format(mergeByUser.count()))
      // anaysis users usage about phones, return most probable one
      val distinctUsed = mergeByUser.map { case (user, usedLst) => (user, distinctUsage(usedLst)) }
      // merged output
      distinctUsed.flatMap { case (user, lst) => lst.map(x => "%s|%s".format(user, x.toString)) }
        .repartition(16).saveAsTextFile("/pnrs/user_with_device_related")
    }
    // Simply static
    val unique = sc.textFile("/pnrs/user_with_device_related").map(_.split('|'))
    val sec2day: Int => String =
      sec => new SimpleDateFormat("yyyyMMdd").format(sec * 1000L)
    val days = unique.map(_.apply(1).toInt).map(sec2day).distinct().collect().sorted
    println("[INFO] distinct days %d".format(days.length))
    println(days.mkString("|"))
    val uNum = unique.map(_.head).distinct().count()
    val dNum = unique.map(_.last).distinct().count()
    println("[INFO] remains %d users and %d devices".format(uNum, dNum))

    val f1 = new FileWriter(new File("./raw_used_device_cont"))
    val mergeByUser = unique.map(ps => (ps(0), ps(2)))
        .aggregateByKey(Set[String]())(_ + _, _ ++ _)
        .map(_._2.size).map(x => (x, 1)).reduceByKey(add)
    mergeByUser.sortBy(_._1).map{case (cont, time) => "%d,%d\n".format(cont, time)}
        .collect().foreach(f1.write)
    f1.close()
    val f2 = new FileWriter(new File("./raw_using_users_cont"))
    val mergeByDevice = unique.map(ps => (ps(2), ps(0)))
      .aggregateByKey(Set[String]())(_ + _, _ ++ _)
      .map(p => (p._1, p._2.size)).sortBy(-_._2)
    mergeByDevice.map{case (name, cont) => "%s,%d\n".format(name, cont)}
        .collect().foreach(f2.write)
    f2.close()
  }

}
