"""visual.py"""
import pandas as pd
import matplotlib.pyplot as plt

SFCRIME = pd.read_csv("data/SF-police-data-2016.csv")
SFDISTRICTS = pd.read_csv("data/SF_Police_Districts.csv", index_col='PdDistrict')

print(SFCRIME["Category"].value_counts())

#Finding the Total crimes

CRIME_DISTCAT = pd.crosstab(index=SFCRIME["PdDistrict"],
                            columns=SFCRIME["Category"])

CRIME_DISTCAT['Total'] = CRIME_DISTCAT.apply(sum, axis=1)
SFCRIME_DISTRICTS = pd.concat([CRIME_DISTCAT, SFDISTRICTS], axis=1)

#Finding the Total Nonviolent crimes

CRIME_DISTCAT_NON_VIOLENT = pd.crosstab(index=SFCRIME["PdDistrict"],
	                                       columns=SFCRIME["Category"])

#According to the Prison Policy Initiative (prisonpolicy.org) the following are "violent" crimes:
del CRIME_DISTCAT_NON_VIOLENT["ASSAULT"]
del CRIME_DISTCAT_NON_VIOLENT["ROBBERY"]
del CRIME_DISTCAT_NON_VIOLENT["SEX OFFENSES, FORCIBLE"]

CRIME_DISTCAT_NON_VIOLENT['Total'] = CRIME_DISTCAT_NON_VIOLENT.apply(sum, axis=1)
SFCRIME_DISTRICTS_NON_VIOLENT = pd.concat([CRIME_DISTCAT_NON_VIOLENT, SFDISTRICTS], axis=1)

SFCRIME_DISTRICTS_NON_VIOLENT["Nonviolent"] = CRIME_DISTCAT_NON_VIOLENT["Total"]

#Graphing the data

GRAPH = pd.concat([SFCRIME_DISTRICTS["Total"], SFCRIME_DISTRICTS_NON_VIOLENT["Nonviolent"]], axis=1)
#Pandas Concat is my new method https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html
GRAPH.plot.bar()
plt.xlabel('District')
plt.ylabel('Crimes Committed')
plt.title('Total Crimes Committed vs Nonviolent Crimes Committed in SF')
plt.tight_layout()
plt.show()

#150,608 Crimes Total (all districts combined)
#Violent Crimes: Assault (13590), Robbery (3300), Sex Offenses (943) -- 17833
#11% of crimes are violent, 89% nonviolent



#Second graph is unrelated to first



#Finding the per capita and population density crime rates

CRIME_DISTCAT = pd.crosstab(index=SFCRIME["PdDistrict"],
                            columns=SFCRIME["Category"])

CRIME_DISTCAT['Total'] = CRIME_DISTCAT.apply(sum, axis=1)
SFCRIME_DISTRICTS = pd.concat([CRIME_DISTCAT, SFDISTRICTS], axis=1)

SFCRIME_DISTRICTS['Per Capita'] = SFCRIME_DISTRICTS['Total'] / SFDISTRICTS['Population']
#SFCRIME_DISTRICTS['Per Area'] = SFCRIME_DISTRICTS['Total'] / SFDISTRICTS['Land Mass']
SFCRIME_DISTRICTS['Pop Density'] = (SFCRIME_DISTRICTS['Total'] /
								                            (SFDISTRICTS['Population'] /
								                            	SFDISTRICTS['Land Mass']))


#Fiding the per capita and poopulation density crime rates without considering theft

CRIME_DISTCAT_NO_THEFT = pd.crosstab(index=SFCRIME["PdDistrict"],
                                     columns=SFCRIME["Category"])

del CRIME_DISTCAT_NO_THEFT["LARCENY/THEFT"]

CRIME_DISTCAT_NO_THEFT['Total'] = CRIME_DISTCAT_NO_THEFT.apply(sum, axis=1)
SFCRIME_DISTRICTS_NO_THEFT = pd.concat([CRIME_DISTCAT_NO_THEFT, SFDISTRICTS], axis=1)

SFCRIME_DISTRICTS_NO_THEFT['Per Capita w/o Theft'] = (SFCRIME_DISTRICTS_NO_THEFT['Total']
													                                         / SFDISTRICTS['Population'])
# SFCRIME_DISTRICTS_NO_THEFT['Per Area w/o Theft'] = (SFCRIME_DISTRICTS_NO_THEFT['Total']
# 													/ SFDISTRICTS['Land Mass'])
SFCRIME_DISTRICTS_NO_THEFT['Pop Density w/o Theft'] = (SFCRIME_DISTRICTS_NO_THEFT['Total']
													                                          / (SFDISTRICTS['Population']
													                                             / SFDISTRICTS['Land Mass']))


#Creating the graph

GRAPH_WITH_THEFT = SFCRIME_DISTRICTS.filter(['Per Capita', 'Pop Density'])
GRAPH_WITHOUT_THEFT = SFCRIME_DISTRICTS_NO_THEFT.filter(['Per Capita w/o Theft',
														                                           'Pop Density w/o Theft'])
GRAPH_FINAL = pd.concat([GRAPH_WITH_THEFT, GRAPH_WITHOUT_THEFT], axis=1)
GRAPH_FINAL.plot.bar()
plt.xlabel('District')
plt.ylabel('Crime Rate')
plt.title('Crime Rates in Each San Francisco District')
plt.tight_layout()
plt.show()



#All of the code I'm not using:
#I was previously trying to find a correlation between days of the week where
#prostitution was reported and days of the week where loitering was reportted,
#but that did not work out.


# Let's get a summary of the data.
#SFCRIME.info()
# you get the same if you: print(SFCRIME)

# print(SFCRIME.head())

# print(SFCRIME["DayOfWeek"].value_counts())

# gby = SFCRIME.groupby(["PdDistrict"])
# with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
# 	print(gby.count())

# for key, value in gby:
# 	print("Key: " + key + " Value:" + value)


# tender = SFCRIME[SFCRIME["PdDistrict"] == 'TENDERLOIN']
# south = SFCRIME[SFCRIME["PdDistrict"] == 'SOUTHERN']
# tender_cats = tender["Category"]
# south_cats = south["Category"]   #Change back eventually
# # print(tender_cats)
# # print(south_cats)
# tender_south_cats = pd.crosstab(SFCRIME["Category"], SFCRIME["PdDistrict"])
# print(tender_south_cats)

# tender_south_cats = pd.concat([tender_cats, south_cats])
# print(tender_south_cats)


# loit = (SFCRIME[SFCRIME['Category'] == 'LOITERING'])
# pros = (SFCRIME[SFCRIME['Category'] == 'PROSTITUTION'])

# loit_pros = pd.concat([loit, pros])


# nor_loit = (northern[northern['Category'] == 'LOITERING'])
# #print(nor_loit["DayOfWeek"].value_counts())
# print(nor_loit)
# nor_pros = (northern[northern['Category'] == 'PROSTITUTION'])
# #print(nor_pros["DayOfWeek"].value_counts())
# print(nor_pros)

# print(SFCRIME_DISTRICTS['per_capita'])
# print(SFCRIME_DISTRICTS['per_area'])
# print(SFCRIME_DISTRICTS['pop_density'])

# #Let's plot the top 10 with a bar chart:
# plt.figure()
# data1 = tender_cats
# data1.plot(kind = 'bar')
# data2 = south_cats
# data2.plot(kind = 'bar')
# #plt.show()
# # Let's do some styling:
# data1.plot(kind = 'bar')
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Title')
# #plt.text(5, 600, "Text")
# #plt.axis([40, 160, 0, 0.03]) # xmin, xmax, ymin, ymax
# #plt.grid(True)

# plt.show()

# We can also have more than one data set on the figure.
#	Lets compare Denver and Vegas...
# plot1 = plt.bar(tender_cats.keys(),
# 					tender_cats)
# # plot2 = plt.bar(south_cats,
# #                     SFCRIME["Category"],
# #                     c='b', label='South')
# plt.legend(handles=[plot1])
# plt.xlabel('Minutes delayed due to Weather')
# plt.ylabel('Total Minutes Delayed')
# plt.title('Vegas vs Denver Delays')
# plt.show()


#THIS ONE WORKS
# SFCRIME_DISTRICTS['per_capita'].plot.bar(color = 'blue')
# SFCRIME_DISTRICTS['pop_density'].plot.bar(color = 'red')
# SFCRIME_DISTRICTS_NO_THEFT['per_capita'].plot.bar(color = 'green')
# SFCRIME_DISTRICTS_NO_THEFT['pop_density'].plot.bar(color = 'black
#print(type(SFCRIME_DISTRICTS))
# print(type(SFCRIME_DISTRICTS['per_capita']))
# print(type(SFCRIME_DISTRICTS['pop_density']))

# series = pd.DataFrame(dict('per_capita' = SFCRIME_DISTRICTS['per_capita'],
# 										  'pop_density' = SFCRIME_DISTRICTS['pop_density']))
# series.plot.bar()

# result = result.(SFCRIME_DISTRICTS['pop_density'])
# result = result.join(SFCRIME_DISTRICTS_NO_THEFT['per_capita'])
# result = result.join(SFCRIME_DISTRICTS_NO_THEFT['pop_density'])

# result.plot.bar()

#print(type(tender_south_cats)) ##THIS COULD BE THE FUNCTION I DIDNT KNOW

# # data to plot
# n_groups = 4
# means_frank = (90, 55, 40, 65)
# means_guido = (85, 62, 54, 20)

# create plot
# fig, ax = plt.subplots()
# index = np.arange(len(SFCRIME["Category"]))
# bar_width = 0.35
# opacity = 0.8


# rects1 = plt.bar(index, tender_cats, bar_width,
#                  alpha = opacity,
#                  color = 'b',
#                  label = 'Tender')

# rects2 = plt.bar(index + bar_width, south_cats, bar_width,
#                  alpha = opacity,
#                  color = 'g',
#                  label = 'South')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Title')
# # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
# plt.legend()
# plt.show()




#NOTES

# #population density
# #what kinds of crimes are happening in these two districts and does that make a difference?
# Or do they have the same kinds of crime?

#Max category of crime for each area
#Stack the with theft and without theft for per capita and pop density
