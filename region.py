central = ['Bishan','Bukit Merah','Geylang','Kallang','Marine Parade','Queenstown','Southern Islands','Toa Payoh','KALLANG/WHAMPOA','CENTRAL AREA']
central = [x.upper() for x in central]
east = ['Bedok','Changi','Geylang','Changi Bay','Paya Lebar','Pasir Ris','Tampines']
east = [x.upper() for x in east]
north = ['Central Water Catchment','Lim Chu Kang','Mandai','Sembawang','Simpang','Sungei Kadut','Sungei Kadut','Yishun']
north = [x.upper() for x in north]
northeast = ['Ang Mo Kio','Hougang','North-Eastern Islands','Punggol','Seletar','Sengkang','Serangoon']
northeast = [x.upper() for x in northeast]
west = ['Bukit Batok','Bukit Panjang','BUKIT TIMAH','Boon Lay','Pioneer','Choa Chu Kang','Clementi','Jurong East','Jurong West','Tengah',
'Tuas','Western Islands','Western Water Catchment','Benoi','Ghim Moh','Gul','Pandan Gardens','Jurong Island','Kent Ridge',
'Nanyang','Pioneer','Pasir Laba','Teban Gardens','Toh Tuck','Tuas South','West Coast','WOODLANDS']
west = [x.upper() for x in west]

region_d = {}
region_d['central']=central
region_d['east']=east
region_d['west']=west
region_d['northeast']=northeast 
region_d['north']=north
print(region_d)