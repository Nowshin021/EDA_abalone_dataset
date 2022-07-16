#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv('abalone.csv')

df.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
#%%
#Skewness and kurtosis
fig, axes = plt.subplots(1, 3, figsize=(20,6))
sns.set_style("darkgrid")
fig.suptitle('Highest and lowest skewness and kurtosis of the dataset : ',fontsize=12)

##Skewness of Height : 
height_skew=df['Height'].skew()
height_skew="{0:.3f}".format(height_skew)
result2="right skewed "+"("+ str(height_skew) +")"


#kurtosis of height :

height_kurt=df['Height'].kurt()
height_kurt = "{0:.3f}".format(height_kurt)

result2_kurt = "leptokurtic "+ "("+ str(height_kurt) +")"

ax1=sns.histplot(df['Height'], ax=axes[1], bins=80, kde=True, color='g')

ax1.set(title="Distribution of height")
ax1.set_ylabel("Frequency", size=12)
#xy==x and y co-ordinates of the arrow
ax1.annotate(result2+"\n"+result2_kurt, xy=(.3, 70),color = 'g',  xycoords='data',
            xytext=(0.2, .8), textcoords='axes fraction',
            fontsize = 14, arrowprops = dict(arrowstyle = '->',color='black',
             connectionstyle = "arc3, rad = -0.3")
            )



length_skew=df['Length'].skew()
length_skew="{0:.3f}".format(length_skew)
result1="left skewed "+"(" +str(length_skew)+")"

ax2=sns.histplot(df['Length'], ax=axes[0], bins=80, kde=True,color='b')
ax2.set(title="Distribution of Length")
ax2.set_ylabel("Frequency", size=12)
#xy==x and y co-ordinates of the arrow
ax2.annotate(result1, xy=(.3, 70),color = 'b',  xycoords='data',
            xytext=(0.1, .8), textcoords='axes fraction',
            fontsize = 14, arrowprops = dict(arrowstyle = '->',color='black',
             connectionstyle = "arc3, rad = 0.3")
            )


#lowest kurtosis:
diameter_kurt=df['Diameter'].kurt()
diameter_kurt = "{0:.3f}".format(diameter_kurt)

result3_kurt = "platykurtic "+ "("+ str(diameter_kurt) +")"

ax3=sns.histplot(df['Diameter'], ax=axes[2], bins=50, kde=True, color='#cc0000')
ax3.set(title="Distribution of Diameter")
ax3.set_ylabel("Frequency", size=12)
#xy==x and y co-ordinates of the arrow
ax3.annotate(result3_kurt, xy=(.2, 50),color = '#cc0000',  xycoords='data',
            xytext=(0.1, .8), textcoords='axes fraction',
            fontsize = 14, arrowprops = dict(arrowstyle = '->',color='black',
             connectionstyle = "arc3, rad = 0.3")
            )
plt.savefig("distribution.png",dpi=72)
plt.show()

print(df['Length'].mean())

#%%
#gridsize: the number of hexagons in the x-direction and the y-direction
#output analysis based on length and diameter

fig, axes = plt.subplots(1 ,2, figsize=(16,8))
fig.suptitle('Hexbin plots to show ther relationship  of Rings(age) with respect to Diameter and Length the dataset : ',fontsize=12)
ax=axes[0]
ax.hexbin(df["Length"], df["Rings"], gridsize=25,cmap=plt.cm.Greens_r)
ax.set(xlabel='Length', ylabel='Number of rings')
ax.set_title("Hexbin plot of Length vs Rings")


ax=axes[1]
ax.hexbin(df["Diameter"], df["Rings"], gridsize=25,cmap=plt.cm.Blues_r)
ax.set(xlabel='Diameter', ylabel='Number of rings')
ax.set_title("Hexbin plot of Diameter vs Rings")
plt.savefig("hexbins.png",dpi=72)
plt.show()

#%%
#correlation max and min
plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
sns.scatterplot(x=df['Shucked weight'], y=df['Rings'])
plt.title("Minimum Correlation")

plt.subplot(2,2,2)
sns.scatterplot(x=df['Length'], y=df['Diameter'])
plt.title("Maximum Correlation")

plt.savefig("correlation.png",dpi=72)
plt.show()

#%%
#output analysis based on gender
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

m = df[['Sex', 'Rings']].query('Sex == "M"')
f = df[['Sex', 'Rings']].query('Sex == "F"')
i = df[['Sex', 'Rings']].query('Sex == "I"')

# plotting both distibutions on the same figure
fig = sns.kdeplot(m['Rings'], label="Male", color="r")
fig = sns.kdeplot(f['Rings'], label="Female", color="b")
fig = sns.kdeplot(i['Rings'], label="Infant", color="g")
plt.legend(loc="upper right")
plt.xlabel("Rings")

plt.savefig("density.png",dpi=72)
plt.show()

#%%
sex_m_length= df[['Sex','Length']].query('Sex=="M"')
sex_m_length_without_sex=sex_m_length['Length']
m_length_median=sex_m_length_without_sex.median()

sex_f_length= df[['Sex','Length']].query('Sex=="F"')
sex_f_length_without_sex=sex_f_length['Length']
f_length_median=sex_f_length_without_sex.median()


sex_i_length= df[['Sex','Length']].query('Sex=="I"')
sex_i_length_without_sex=sex_i_length['Length']
i_length_median=sex_i_length_without_sex.median()

#%%
sex_m_height= df[['Sex','Height']].query('Sex=="M"')
sex_m_height_without_sex=sex_m_height['Height']
m_height_median=sex_m_height_without_sex.median()


sex_f_height= df[['Sex','Height']].query('Sex=="F"')
sex_f_height_without_sex=sex_f_height['Height']
f_height_median= sex_f_height_without_sex.median()

sex_i_height= df[['Sex','Height']].query('Sex=="I"')
sex_i_height_without_sex=sex_i_height['Height']
i_height_median=sex_i_height_without_sex.median()

#%%
sex_m_wweight= df[['Sex','Whole weight']].query('Sex=="M"')
sex_m_wweight_without_sex=sex_m_wweight['Whole weight']
m_wweight_median=sex_m_wweight_without_sex.median()

sex_f_wweight= df[['Sex','Whole weight']].query('Sex=="F"')
sex_f_wweight_without_sex=sex_f_wweight['Whole weight']
f_wweight_median=sex_f_wweight_without_sex.median()


sex_i_wweight= df[['Sex','Whole weight']].query('Sex=="I"')
sex_i_wweight_without_sex=sex_i_wweight['Whole weight']
i_wweight_median=sex_i_wweight_without_sex.median()

#%%
sex_m_diameter= df[['Sex','Diameter']].query('Sex=="M"')
sex_m_diameter_without_sex=sex_m_diameter['Diameter']
m_diameter_median=sex_m_diameter_without_sex.median()

sex_f_diameter= df[['Sex','Diameter']].query('Sex=="F"')
sex_f_diameter_without_sex=sex_f_diameter['Diameter']
f_diameter_median=sex_f_diameter_without_sex.median()


sex_i_diameter= df[['Sex','Diameter']].query('Sex=="I"')
sex_i_diameter_without_sex=sex_i_diameter['Diameter']
i_diameter_median=sex_i_diameter_without_sex.median()

#%%
#figure analysis based on gender
plt.figure (figsize=(15,20), dpi = 80)
colors = ['pink', 'lightblue', 'lightgreen']

column=np.array([sex_m_length_without_sex,sex_f_length_without_sex,sex_i_length_without_sex],dtype='object')
plt.subplot(2,2,1)
box1=plt.boxplot(column,labels=['Male', 'Female','Infant'],patch_artist=True,notch=True)
plt.grid()
plt.xlabel('Sex')
plt.ylabel('Length')


plt.annotate(r'$median=%f$'%(m_length_median), xy = (1, m_length_median), xycoords = 'data', xytext = (+20, +60), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(f_length_median), xy = (2, f_length_median), xycoords = 'data', xytext = (+20, +50), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(i_length_median), xy = (3, i_length_median), xycoords = 'data', xytext = (-50, +60), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))


  
column=np.array([sex_m_height_without_sex,sex_f_height_without_sex,sex_i_height_without_sex],dtype='object')
plt.subplot(2,2,2)
box2=plt.boxplot(column,labels=['Male', 'Female','Infant'],patch_artist=True,notch=True)
plt.grid()
plt.xlabel('Sex')
plt.ylabel('Height')

plt.annotate(r'$median=%f$'%(m_height_median), xy = (1, m_height_median), xycoords = 'data', xytext = (+10, +60), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(f_height_median), xy = (2, f_height_median), xycoords = 'data', xytext = (+10, +80), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(i_height_median), xy = (3, i_height_median), xycoords = 'data', xytext = (-50, +65), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))

    
column=np.array([sex_m_wweight_without_sex,sex_f_wweight_without_sex,sex_i_wweight_without_sex],dtype='object')
plt.subplot(2,2,3)
box3=plt.boxplot(column,labels=['Male', 'Female','Infant'],patch_artist=True,notch=True)
plt.grid()
plt.xlabel('Sex')
plt.ylabel('Whole Weight')

plt.annotate(r'$median=%f$'%(m_wweight_median), xy = (1, m_wweight_median), xycoords = 'data', xytext = (-50, +70), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(f_wweight_median), xy = (2, f_wweight_median), xycoords = 'data', xytext = (-50, +80), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(i_wweight_median), xy = (3, i_wweight_median), xycoords = 'data', xytext = (-50, +75), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))

   
column=np.array([sex_m_diameter_without_sex,sex_f_diameter_without_sex,sex_i_diameter_without_sex],dtype='object')
plt.subplot(2,2,4)
box4=plt.boxplot(column,labels=['Male', 'Female','Infant'],patch_artist=True,notch=True)
plt.grid()
plt.xlabel('Sex')
plt.ylabel('Diameter')

plt.annotate(r'$median=%f$'%(m_diameter_median), xy = (1, m_diameter_median), xycoords = 'data', xytext = (-40, +70), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(f_diameter_median), xy = (2, f_diameter_median), xycoords = 'data', xytext = (-40, +70), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))
plt.annotate(r'$median=%f$'%(i_diameter_median), xy = (3, i_diameter_median), xycoords = 'data', xytext = (-50, +75), textcoords = 'offset points',
             fontsize = 10, arrowprops = dict(arrowstyle= '->', connectionstyle = "arc3, rad = .2"))

for box in (box1, box2, box3, box4):
  for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig("pattern.png",dpi=72)
plt.show()

#%%
m_sweight_rings= df[['Sex','Shucked weight','Rings']].query('Sex=="M"')
m_sweight_rings_withoutSex=m_sweight_rings[['Shucked weight','Rings']]


f_sweight_rings= df[['Sex','Shucked weight','Rings']].query('Sex=="F"')
f_sweight_rings_withoutSex=f_sweight_rings[['Shucked weight','Rings']]


i_sweight_rings= df[['Sex','Shucked weight','Rings']].query('Sex=="I"')
i_sweight_rings_withoutSex=i_sweight_rings[['Shucked weight','Rings']]

#%%
m_wweight_rings= df[['Sex','Whole weight','Rings']].query('Sex=="M"')
m_wweight_rings_withoutSex=m_wweight_rings[['Whole weight','Rings']]


f_wweight_rings= df[['Sex','Whole weight','Rings']].query('Sex=="F"')
f_wweight_rings_withoutSex=f_wweight_rings[['Whole weight','Rings']]


i_wweight_rings= df[['Sex','Whole weight','Rings']].query('Sex=="I"')
i_wweight_rings_withoutSex=i_wweight_rings[['Whole weight','Rings']]

#%%
m_shweight_rings= df[['Sex','Shell weight','Rings']].query('Sex=="M"')
m_shweight_rings_withoutSex=m_shweight_rings[['Shell weight','Rings']]


f_shweight_rings= df[['Sex','Shell weight','Rings']].query('Sex=="F"')
f_shweight_rings_withoutSex=f_shweight_rings[['Shell weight','Rings']]


i_shweight_rings= df[['Sex','Shell weight','Rings']].query('Sex=="I"')
i_shweight_rings_withoutSex=i_shweight_rings[['Shell weight','Rings']]

#%%
m_vweight_rings= df[['Sex','Viscera weight','Rings']].query('Sex=="M"')
m_vweight_rings_withoutSex=m_vweight_rings[['Viscera weight','Rings']]


f_vweight_rings= df[['Sex','Viscera weight','Rings']].query('Sex=="F"')
f_vweight_rings_withoutSex=f_vweight_rings[['Viscera weight','Rings']]


i_vweight_rings= df[['Sex','Viscera weight','Rings']].query('Sex=="I"')
i_vweight_rings_withoutSex=i_vweight_rings[['Viscera weight','Rings']]

#%%
#output analysis based on weight
plt.figure (figsize=(20,10), dpi = 80)
plt.subplot(2,2,1)
plt.scatter(m_sweight_rings_withoutSex['Shucked weight'],m_sweight_rings_withoutSex['Rings'], color='red', marker='o',label='Male')
plt.scatter(f_sweight_rings_withoutSex['Shucked weight'],f_sweight_rings_withoutSex['Rings'], color='green', marker='*',label='Female')
plt.scatter(i_sweight_rings_withoutSex['Shucked weight'],i_sweight_rings_withoutSex['Rings'], color='purple', marker='v',label='Infant')

plt.legend(loc="upper right")
plt.grid()
plt.xlabel('Shucked weight')
plt.ylabel('Rings/Age')

plt.subplot(2,2,2)
plt.scatter(m_wweight_rings_withoutSex['Whole weight'],m_wweight_rings_withoutSex['Rings'], color='red', marker='o',label='Male')
plt.scatter(f_wweight_rings_withoutSex['Whole weight'],f_wweight_rings_withoutSex['Rings'], color='green', marker='*',label='Female')
plt.scatter(i_wweight_rings_withoutSex['Whole weight'],i_wweight_rings_withoutSex['Rings'], color='purple', marker='v',label='Infant')

plt.legend(loc="upper right")
plt.grid()
plt.xlabel('Whole weight')
plt.ylabel('Rings/Age')


plt.subplot(2,2,3)
plt.scatter(m_shweight_rings_withoutSex['Shell weight'],m_shweight_rings_withoutSex['Rings'], color='red', marker='o',label='Male')
plt.scatter(f_shweight_rings_withoutSex['Shell weight'],f_shweight_rings_withoutSex['Rings'], color='green', marker='*',label='Female')
plt.scatter(i_shweight_rings_withoutSex['Shell weight'],i_shweight_rings_withoutSex['Rings'], color='purple', marker='v',label='Infant')

plt.legend(loc="upper right")
plt.grid()
plt.xlabel('Shell weight')
plt.ylabel('Rings/Age')

plt.subplot(2,2,4)
plt.scatter(m_vweight_rings_withoutSex['Viscera weight'],m_vweight_rings_withoutSex['Rings'], color='red', marker='o',label='Male')
plt.scatter(f_vweight_rings_withoutSex['Viscera weight'],f_vweight_rings_withoutSex['Rings'], color='green', marker='*',label='Female')
plt.scatter(i_vweight_rings_withoutSex['Viscera weight'],i_vweight_rings_withoutSex['Rings'], color='purple', marker='v',label='Infant')

plt.legend(loc="upper right")
plt.grid()
plt.xlabel('Viscera weight')
plt.ylabel('Rings/Age')

plt.savefig("weightnalysis.png",dpi=72)
plt.show()

#%%
