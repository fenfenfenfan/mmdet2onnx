import numpy as np

def generate_new_category():
    cat = [0 for i in range(7440)]
    cat = np.array(cat)
    cat[7290:7440]=1   
    
    cat[2185:2202]=2
    cat[4368:4411]=2
    cat[4460:4496]=2
    cat[6798:6818]=2
    cat[6849:6878]=2

    cat[2202:2250]=3
    cat[4346:4368]=3
    cat[4411:4460]=3
    cat[7048:7062]=3
    cat[7114:7124]=3

    cat[6076:6099]=4
    cat[6137:6211]=4
    cat[7124:7177]=4

    cat[5071:5131]=5
    cat[5202:5257]=5
    cat[5342:5377]=5

    cat[1716:1745]=6
    cat[1774:1895]=6

    cat[988:1094]=7
    cat[1144:1188]=7

    cat[1094:1144]=8
    cat[1188:1236]=8
    cat[1313:1341]=8
    cat[6367:6391]=8

    cat[2371:2487]=9
    cat[4855:4889]=9

    cat[962:988]=10
    cat[1915:1930]=10
    cat[1977:1996]=10
    cat[6391:6411]=10
    cat[6878:6888]=10
    cat[6910:6970]=10

    cat[1236:1281]=11
    cat[1895:1915]=11
    cat[1930:1977]=11
    cat[6888:6910]=11

    cat[345:404]=12
    cat[1565:1628]=12
    cat[1689:1716]=12

    cat[547:607]=13
    cat[1628:1689]=13
    cat[1745:1774]=13

    cat[239:345]=14
    cat[6767:6783]=14
    cat[6818:6849]=14

    cat[2609:2670]=15
    cat[4524:4613]=15

    cat[52:73]=16
    cat[893:932]=16
    cat[2250:2289]=16
    cat[6576:6608]=16
    cat[6783:6798]=16
    cat[6574]=16

    cat[628:648]=17
    cat[704:747]=17
    cat[804:844]=17
    cat[6608:6621]=17
    cat[6691:6721]=17
    cat[6575]=17
    cat[6685]=17
    cat[6688]=17

    cat[0:30]=18
    cat[73:193]=18

    cat[932:962]=19
    cat[1341:1394]=19
    cat[1455:1509]=19
    cat[6251:6264]=19

    cat[30:52]=20
    cat[193:239]=20
    cat[844:893]=20
    cat[6650]=20
    cat[6652:6671]=20
    cat[6686:6688]=20
    cat[6690]=20
    cat[6721:6730]=20

    cat[607:628]=21
    cat[648:704]=21
    cat[747:804]=21
    cat[6671:6685]=21
    cat[6651]=21

    cat[404:547]=22
    cat[6643:6650]=22

    cat[1996:2119]=23
    cat[4078:4105]=23

    cat[4613:4684]=24
    cat[4704:4761]=24
    cat[7177:7198]=24

    cat[5012:5071]=25
    cat[5139:5202]=25
    cat[5314:5342]=25

    cat[2119:2185]=26
    cat[4291:4346]=26
    cat[6047:6076]=26

    cat[4157:4165]=27
    cat[4199:4223]=27
    cat[4266:4291]=27
    cat[7198:7290]=27

    cat[2552:2609]=28
    cat[3505:3566]=28
    cat[3621:3651]=28
    cat[6730:6732]=28

    cat[2347:2371]=29
    cat[5131:5139]=29
    cat[5540:5559]=29
    cat[6486:6505]=29
    cat[6529:6574]=29
    cat[6732:6767]=29

    cat[4019:4078]=30
    cat[4105:4157]=30
    cat[4165:4199]=30


    cat[4223:4266]=31
    cat[4684:4704]=31
    cat[4761:4802]=31
    cat[6464:6486]=31
    cat[6505:6529]=31

    cat[3248:3276]=32
    cat[3304:3354]=32
    cat[3384:3444]=32
    cat[6433:6438]=32

    cat[2670:2703]=33
    cat[3058:3175]=33

    cat[3276:3304]=34
    cat[3915:4019]=34
    cat[6438:6456]=34

    cat[2703:2731]=35
    cat[2923:3031]=35
    cat[6622:6635]=35

    cat[2731:2770]=36
    cat[2839:2874]=36
    cat[3175:3189]=36
    cat[6635:6643]=36
    cat[7062:7114]=36
    cat[6621]=36

    cat[3566:3621]=37
    cat[3651:3717]=37
    cat[4496:4524]=37

    cat[3189:3248]=38
    cat[3354:3384]=38
    cat[3444:3505]=38

    cat[2770:2839]=39
    cat[2874:2923]=39
    cat[3031:3058]=39
    cat[7043:7048]=39

    cat[5666:5685]=40
    cat[5741:5745]=40
    cat[5746:5778]=40
    cat[5835:5871]=40
    cat[6972:6987]=40
    cat[6970]=40
    cat[7002:7043]=40

    cat[5377:5399]=41
    cat[5462:5477]=41
    cat[5506:5540]=41
    cat[6211:6235]=41
    cat[6322:6345]=41
    cat[6411:6433]=41
    cat[6456:6464]=41

    cat[3717:3787]=42
    cat[3818:3861]=42
    cat[6307:6322]=42
    cat[6345:6367]=42

    cat[2487:2552]=43
    cat[3787:3818]=43
    cat[3861:3915]=43

    cat[5936:6047]=44
    cat[6099:6137]=44

    cat[4802:4855]=45
    cat[4967:4998]=45
    cat[5257:5314]=45
    cat[6298:6305]=45

    cat[4889:4967]=46
    cat[4998:5012]=46
    cat[6235:6251]=46
    cat[6264:6298]=46
    cat[6305:6307]=46

    cat[2289:2347]=47
    cat[5399:5462]=47
    cat[5477:5506]=47

    cat[1281:1313]=48
    cat[1394:1455]=48
    cat[1509:1565]=48

    cat[5713:5741]=49
    cat[5778:5835]=49
    cat[5871:5936]=49

    cat[5559:5666]=50
    cat[5685:5713]=50
    cat[6987:7002]=50

    cat[5745]=99
    cat[6689]=99
    cat[6971]=99

    cat = cat.tolist()
    return cat

def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] == item]
# cat = generate_new_category()
# other=get_index(cat,99)

