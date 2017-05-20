
# coding: utf-8

# <div style="color:red">!!! N.B. : Toutes les informations ci-dessous sont extraites à partir du fichier fourni dans le livrable et datant du 09/03/2017. <br> 
# Aujourd'hui, le fichier a plus que doublé de volume, on est passé d'un peu moins de 140 000 enregistrements à plus de 326 000 grâce à l'apport d'enregistrements principalement américains. </div>

# <h2>1- Description des données</h2>
# <h5>Importation des données et des packages</h5>

# In[122]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')

# fichier des données, adresse à adapter
dfOriginel = pd.read_csv('fr.openfoodfacts.org.products.csv', sep='\t')


# <h5>Volume des données</h5>

# In[123]:

print("Il y a {0} enregistrements et {1} variables".format(dfOriginel.shape[0], dfOriginel.shape[1]))


# <h5>Doublons</h5>

# La gestion automatique des doublons pose ici problème.<br>
# On pourrait supprimer automatiquement tous les doublons sans se poser de question :

# In[124]:

print(dfOriginel.shape)
df2 = dfOriginel.drop_duplicates(subset='code', keep='last')
print(df2.shape)


# On voit que cela a supprimé 34 enregistrements. Le problème est de vérifier si ce sont des doublons réels !?

# In[125]:

codes = dfOriginel.code.value_counts()
print(codes[codes>1])
dfOriginel.set_index('code').index.get_duplicates()


# Et lorsqu'on y regarde de plus près, on s'aperçoit que par exemple les 23 enregistrements qui n'ont pas de code (NaN) sont quand même des produits dont nombre d'informations sont renseignées. 
# Les cas où les enregistrements ont des codes identiques sont plus problématiques car tous les cas sont représentés, porter sur des articles différents, ou sur les mêmes articles mais avec des informations différentes. Il devient dans ce cas difficile de choisir arbitrairement quel enregistrement garder.  
# On pourrait utiliser les photographies des articles pour aider à voir si deux articles sont identiques, mais parfois pour une même marque il y a une déclinaison de design qui fait qu'il est difficile sans avoir le produit en main de déterminer s'il y a doublon ou non.
# Il est donc préférable de considérer que ces doublons (sur le code) n'en sont pas puisque ces enregistrements possèdent des informations différentes. Ils sont donc maintenus. 

# <h5>Données manquantes</h5> (pourcentage de NaN pour chaque variable, ordonné par ordre croissant)

# In[126]:

dfNanOriginel = 100*(1-dfOriginel.count()/dfOriginel.shape[0])

print(dfNanOriginel.sort_values())  


# In[127]:

X = (100-dfNanOriginel).sort_values()
plt.bar(range(len(X)), X)
plt.xlabel("indice des variables")
plt.ylabel("%")
plt.title(u'% ordonné de remplissage des variables')


# On peut mieux visualiser la proportion de NaN par variable avec ce graphique:

# In[128]:

plt.figure(figsize=(10, 25))
dfOriginel.isnull().mean(axis=0).plot.barh()
plt.title("Proportion de NaN pour chaque variable")


# <h5>Quelques informations sur le remplissage des variables :</h5>

# In[129]:

print("Nombre de variables : {0}".format(dfOriginel.shape[1]))
print("Nombre de variables vides: {0} sur {1}".format(len((np.where(dfNanOriginel == 100.0))[0]), dfOriginel.shape[1]))
print("Nombre de variables non vides remplies à moins de 1%: {0} sur {1}".format(len(dfNanOriginel[(dfNanOriginel > 99.0) & (dfNanOriginel < 100.0)]), dfOriginel.shape[1]))
print("Nombre de variables remplies à plus de 90%: {0} sur {1}".format(len((np.where(dfNanOriginel < 10.0))[0]), dfOriginel.shape[1]))

# dataframe des variables décrivant des composants (se terminent par _100g)
dfComposant = pd.DataFrame()
i = 0
while i < dfOriginel.shape[1]:
    if (dfOriginel.dtypes[i] == np.float64) & ("_100g" in dfOriginel.columns[i]):
        dfComposant.insert(dfComposant.shape[1], dfOriginel.columns[i], dfOriginel[dfOriginel.columns[i]])
    i = i + 1
dfNanComposant = 100*(1-dfComposant.count()/dfComposant.shape[0])

print("Nombre de variables composant: {0}".format(dfNanComposant.shape[0]))
print("Nombre de variables composant vides: {0} sur {1}".format(len((np.where(dfNanComposant == 100.0))[0]), dfNanComposant.shape[0]))
print("Nombre de variables composant non vides remplies à moins de 1%: {0} sur {1}".format(len(dfNanComposant[(dfNanComposant > 99.0) & (dfNanComposant < 100.0)]), dfNanComposant.shape[0]))
print("Nombre de variables composant remplies à moins de 4 %: {0} sur {1}".format(len((np.where(dfNanComposant > 96.0))[0]), dfNanComposant.shape[0]))


# <u>Commentaires</u> : <br>
# On remarque d'après ces résultats que la plupart des variables ne sont que très peu renseignées. Par exemple, sur les variables les plus importantes que sont les composants des articles, sur les 98 variables, seules 11 sont renseignées à plus de 4%. 

# In[130]:

print(dfComposant.count().sort_values())


# In[131]:

def make_autopct(values):
    def my_autopct(pct):
        """Customisation de l'affichage des valeurs {p:.2f}% ({v:d}) 
        Arguments:
        values -- nombres de variables pour chaque section
        pct -- pourcentages de variables pour chaque section
        """
        total = sum(values)
        val = int(round(pct*total/100.0))
        if val == 0:
            return ''
        else:
            return '{p:.2f}% ({v:d})'.format(p=pct, v=val)
    return my_autopct

def make_autopct2(values):
    def my_autopct2(pct):
        """Customisation de l'affichage des valeurs {v:d} 
        Arguments:
        values -- nombres de variables pour chaque section
        pct -- pourcentages de variables pour chaque section
        """        
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}'.format(v=val)
    return my_autopct2

nb_enreg = dfOriginel.shape[0]
classement=[]
for i in range(10):
    classement.append(0)
for i in range (dfComposant.count().shape[0]):
    classement[dfComposant.count()[i]/(nb_enreg/10)] += 1

explode = (0.10, 0, 0, 0.1, 0.25, 0.1, 0, 0, 0, 0)

the_grid = GridSpec(2,2)
plt.clf()
plt.figure(figsize=(17,10))
plt.subplot(the_grid[:,0], aspect=1)
_, _, autotexts = plt.pie(classement, explode=explode, autopct=make_autopct(classement), startangle=90, shadow=True, pctdistance=0.5) #'%1.1f%%'
plt.title(u"% de variables numériques (nombre de variables) ayant\n entre . et . % de données renseignées (couleurs)", bbox={'facecolor':'0.8', 'pad':5})
autotexts[0].set_color('white')
autotexts[0].set_fontsize(16)
autotexts[3].set_color('black')
autotexts[3].set_fontsize(16)
autotexts[3].set_position((0.60, 0.10))
autotexts[4].set_color('black')
autotexts[4].set_fontsize(16)
autotexts[4].set_position((0.70, 0.35))
autotexts[5].set_color('black')
autotexts[5].set_fontsize(16)
autotexts[5].set_position((0.30, 0.90))
plt.axis('equal')
plt.annotate('', xy=(0.27, 0.34), xytext=(0.40, 0.15), arrowprops={'facecolor':'black', 'shrink':0.05})
plt.annotate('', xy=(0.40, 0.62), xytext=(0.62, 0.43), arrowprops={'facecolor':'black', 'shrink':0.05})
labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
plt.legend(labels, loc='lower right')
values = [65, 33]
plt.subplot(the_grid[0,1], aspect=1)
_, _, autotexts = plt.pie(values, autopct=make_autopct2(values))
autotexts[0].set_color('white')
autotexts[1].set_color('black')
for i in range(2):
    autotexts[i].set_fontsize(16)
plt.title(u"Nombre de variables numériques ayant\n moins de 1 % de données renseignées", bbox={'facecolor':'0.8', 'pad':5})
plt.legend(['<1% rempli', '>1% rempli'], loc='lower right')
plt.subplot(the_grid[1,1], aspect=1)
values = [87, 11]
_, _, autotexts = plt.pie(values, autopct=make_autopct2(values))
autotexts[0].set_color('white')
autotexts[1].set_color('black')
for i in range(2):
    autotexts[i].set_fontsize(16)
plt.title(u"Nombre de variables numériques ayant\n moins de 4 % de données renseignées", bbox={'facecolor':'0.8', 'pad':5})
plt.legend(['<4% rempli', '>4% rempli'], loc='lower right')
plt.show()


# <u>Commentaires</u> : <br>
# (Les flèches ont un positionnement absolu, elles ne sont pas adaptées à un jeu de données récent où les sections sont différentes.)<br>
# Comme certaines des variables ne sont pas du tout renseignées, sur ce jeu de donnée on peut écarter ces variables puisqu'elles ne participent pas à l'information. 
# Mais qu'en est-il des autres variables, à partir de quel nombre de données peut-on se permettre de les supprimer du jeu afin de diminuer l'excès de complexité inutile (diminuer le nombre de variables) ? Comment comparer l'utilité de la variable montanic-acid_100g qui ne possède qu'un seul enregistrement de renseigné avec celle de la variable energy_100g qui en possède 80580 ? 
# Prenons l'exemple de l'acide montanique dont la seule occurence se trouve dans l'enregistrement d'une huile d'olive bio. Son intérêt sur l'ensemble des 139578 articles est sans doute assez faible, du moins pour cet instantanné, à ce moment ponctuel de renseignement des informations. Mais supposons qu'une personne fasse demain une étude sur cet élément (dont l'esther semble être un additif (E912) dont la toxicité fasse débat) et ajoute dans les données la quantité de cet élément dans une centaine d'huiles différentes. Cette variable prendrait instantanément un grand intérêt pour l'étude d'un groupe particulier d'articles, les huiles. 
# Alors quel seuil de renseignement peut-on choisir ? 10 %, 20%, 30 % ? 
# Peut-on choisir un seuil vis-à-vis de la variance ? En effet si la variance est presque nulle, cela signifie que la distribution de la variable est proche d'une distribution constante, cela n'apporte donc pas d'information discriminante par rapport aux autres variables. Aussi, peut-on choisir un seuil de 0.01, 0.1, ... ? 
# En l'absence de modèle possédant une métrique de performance permettant de comparer l'influence des différents seuils, il semble arbitraire de choisir un seuil de suppression de variable à ce point d'étude du jeu de données. 
# Même si la possibilité que demain une étude porte sur une des variables actuellement non renseignées, on va se contenter pour l'instant de ne pas tenir compte des variables sans aucune donnée. 

# <h5>Quelques remarques sur les données</h5>

# In[132]:

def no_null_objects(_df, columns=None):
    """
    Méthode qui ne sélectionne que les lignes sans NaN pour un ensemble de colonnes
    Arguments:
    _df -- le dataframe contenant les colonnes
    Retour: le dataframe épuré des lignes avec les NaN sur les colonnes données
    """
    if columns is None:
        columns = _df.columns
    return _df[np.logical_not(np.any(_df[columns].isnull().values, axis=1))]

def splitDataFrameList(_df, _target_column, _separator):
    """
    Méthode qui sépare les données pour une colonne donnée
    Arguments:
    _df -- le dataframe contenant la colonne
    _target_column -- la colonne cible
    _separator -- le séparateur de données
    Retour: le nouveau dataframe avec les données séparées
    """
    def splitListToRows(row, row_accumulator, _target_column, _separator):
        split_row = row[_target_column].split(_separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[_target_column] = s[3:] # suppression des 3 premiers caractères "en:" afin de n'avoir que le pays
            row_accumulator.append(new_row)
    new_rows = []
    _df.apply(splitListToRows,axis=1,args = (new_rows,_target_column,_separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

pays_representes = splitDataFrameList(no_null_objects(dfOriginel, ["countries_tags"]), "countries_tags", ",")
pays_nombre = pays_representes["countries_tags"].value_counts()
pays_nombre[:20][::-1].plot.barh()


# On voit ainsi qu'une très grande majorité des articles renseignés sont des articles vendus en France.<br>
# (Avec un jeu de donnée récent, même si les enregistrements français ont augmenté, ils sont largement dépassés par le nombre d'enregistrements américains.)

# <h2>2- Début de feature engineering</h2>

# <h5>Suppression des variables non renseignées dans le dataframe</h5>

# In[133]:

dfOriginelSans100NaN = pd.DataFrame()
co = dfOriginel.count()
i = 0
while i < dfOriginel.shape[1]:
    if co[i] > 0:
        dfOriginelSans100NaN.insert(dfOriginelSans100NaN.shape[1], dfOriginel.columns[i], dfOriginel[dfOriginel.columns[i]])
    else:
        print(dfOriginel.columns[i])
        print(dfOriginel[dfOriginel.columns[i]].dtypes)
    i = i + 1
print("{0} variables supprimées.".format(dfOriginel.shape[1]-dfOriginelSans100NaN.shape[1]))


# <h5>séparation variables numériques / non numériques</h5>

# In[134]:

dfContinues = pd.DataFrame()
dfNonContinues = pd.DataFrame()

i = 0
while i < dfOriginelSans100NaN.shape[1]:
    if dfOriginelSans100NaN.dtypes[i] == np.float64:
        dfContinues.insert(dfContinues.shape[1], dfOriginelSans100NaN.columns[i], dfOriginelSans100NaN[dfOriginelSans100NaN.columns[i]])
    else:
        dfNonContinues.insert(dfNonContinues.shape[1], dfOriginelSans100NaN.columns[i], dfOriginelSans100NaN[dfOriginelSans100NaN.columns[i]])
    i = i + 1

dfNanContinues = 100*(1-dfContinues.count()/dfContinues.shape[0])


# In[135]:

values = [dfContinues.shape[1], dfNonContinues.shape[1]]
_, _, autotexts = plt.pie(values, autopct=make_autopct2(values))
for i in range(2):
    autotexts[i].set_color('white')
    autotexts[i].set_fontsize(16)
plt.title("Separation variables numeriques / non numeriques\n apres suppression variables vides", bbox={'facecolor':'0.8', 'pad':5})
plt.legend(['Numerique', 'Non Numerique'], loc='lower right')


# In[136]:

plt.figure(figsize=(12,4))

plt.subplot(121)

X1 = (100-dfNanOriginel).sort_values()
plt.bar(range(len(X1)), X1)
plt.xlabel("indice des variables")
plt.ylabel("%")
plt.title(u'% ordonné de remplissage des variables')

plt.subplot(122)

X2 = (100-dfNanContinues).sort_values()
plt.bar(range(len(X2)), X2)
plt.xlabel("indice des variables")
plt.ylabel("%")
plt.title(u'% ordonné de remplissage des variables numériques')


# <u>Commentaires</u> :<br>
# On voit ainsi que les variables qui sont bien remplies sont pour la plupart des variables textes (dont certaines sont remplies automatiquement). Les variables numériques ("composantes") sont, à quelques exceptions, très peu remplies, et lorsqu'on étudie le dataset, les variables les plus remplies sont soit des variables "construites" (scores, énergie), soit des variables "générales" (glucides, protéines, lipides) utilisées dans les calculs de diététique, ce qui semble cohérent puisque ce sont des variables qui regroupent des familles d'autres variables pour les calculs, on peut donc les calculer pour plus d'articles.

# <h2>3- Statistiques univariées</h2>

# <h5>Exemple de quelques statistiques pour les 3 premières variables continues</h5>

# In[137]:

dfContinues3 = dfContinues.iloc[:,:15]
print(dfContinues3.describe())


# In[138]:

def graphe_extremes(_coeff):
    """Graphes des % de valeurs extrêmes 
    Arguments:
    _coeff -- coefficient multiplicateur pour déterminer le seuil (moyenne + coeff * écart-type)
    """
    valExtr = dfComposant[dfComposant>dfComposant.mean()+_coeff*dfComposant.std()].count()
    liste_nom_val_extreme = []
    valCo = dfComposant.count()
    for nom in dfComposant.columns:
        if valExtr[nom] > 0:
            liste_nom_val_extreme.append((nom, 100.0/valCo[nom]*valExtr[nom]))

    liste_nom_val_extreme_sorted = sorted(liste_nom_val_extreme, key=lambda x:x[1]) 
    liste_val_extreme_sorted = []
    for i in range(len(liste_nom_val_extreme_sorted)):
        liste_val_extreme_sorted.append(liste_nom_val_extreme_sorted[i][1])

    return liste_val_extreme_sorted

liste_val_extreme_sorted15 = graphe_extremes(1.5)
liste_val_extreme_sorted3 = graphe_extremes(3)

plt.clf()
plt.figure(figsize=(15,4))
plt.subplot(121)
plt.bar(range(len(liste_val_extreme_sorted15)), liste_val_extreme_sorted15)
plt.xlabel("indice des variables")
plt.ylabel("%")
plt.title(u'% ordonné du nombre de valeurs extrêmes (> mean + 1.5std)')
plt.subplot(122)
plt.xlabel("indice des variables")
plt.ylabel("%")
plt.bar(range(len(liste_val_extreme_sorted3)), liste_val_extreme_sorted3)
plt.title(u'% ordonné du nombre de valeurs extrêmes (> mean + 3std)')
plt.show()
    


# <u>Commentaires</u> :<br>
# Que ce soit au-delà de 1.5 ou 3 std, le pourcentage de "valeurs extrêmes" est assez important. Comme le jeu de donnée ne regroupe pas que des catégories spécifiques d'articles, mais la totalité de ce qui peut exister dans toutes les catégories d'aliments possibles, il n'y a pas de raison d'annuler ces valeurs extrêmes (sinon, par exemple, ce serait comme vouloir annuler la valeur salt_100g de l'article "sel de table" parce qu'il n'y a pas beaucoup de sel dans la plupart des articles). L'étude des valeurs extrêmes doit donc se faire en rapport avec les catégories des articles.    

# <h5>Moyennes des variables continues ordonnées afin de mettre en valeur des éventuelles données aberrantes</h5>

# In[139]:

print(dfContinues.mean().sort_values())


# <u>Commentaires</u> : <br>
# Si la moyenne de 1113 pour la variable energy_100g semble valide, on peut s'interroger sur celles des trois dernières variables, proteins_100g, fat_100g et carbohydrates_100g.

# <h5>Utilisation des boxplots afin de repérer d'éventuelles asymétries aberrantes</h5>

# Comme on peut le voir par exemple avec les boxplots des variables oleic-acid_100g et polyols_100g dont les moyennes semblent cohérentes, il ne semble pas y avoir de valeurs aberrantes pour ces variables : 

# In[140]:

def dfBoxPlot(_df, _title):
    """Dessine le boxplot 
    Arguments:
    _df -- dataframe contenant les données initiales
    _title -- titre du boxplot
    """
    plt.clf() 
    colors=dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    boxprops = dict(linewidth=2)
    medianprops = dict(linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    flierprops = dict(marker='o', linestyle='none', markerfacecolor='green', markersize=6)
    ax = _df.plot.box(showmeans=True, showfliers=True, flierprops=flierprops, boxprops=boxprops, medianprops=medianprops, color=colors, meanprops=meanpointprops)
    ax.set_ylabel('Valeurs des variables')
    ax.set_title(_title)
    plt.show()
    
dfBoxPlot(dfOriginel[["oleic-acid_100g", "polyols_100g"]], "Boxplot des variables oleic-acid_100g et polyols_100g")


# Par contre, en ce qui concerne les variables proteins_100g, fat_100g et carbohydrates_100g, on peut voir qu'il y a des valeurs aberrantes qui sont hors de proportion par rapport au reste des données 

# In[141]:

dfBoxPlot(dfOriginel[["proteins_100g", "fat_100g", "carbohydrates_100g"]], "Boxplot des variables proteins_100g, fat_100g et carbohydrates_100g")


# Le problème se situe sur un seul enregistrement pour ces trois variables, celui qui possède comme code 15666666666, et dont la valeur pour ces trois variables est "15666666000". 
# On peut donc considérer que ces valeurs perturbent ces variables, on peut les annuler afin de comparer l'éventuelle modification des moyennes :<br>
# (dans le fichier téléchargé le 16/05/2017, cet enregistrement a disparu..., donc les boxplots pour ces 3 variables sont "normales")

# On voit bien que l'annulation de ces 3 valeurs a rendu les moyennes plus cohérentes et que l'asymétrie a nettement diminué

# In[142]:

# ajout de la variable code comme index afin de pouvoir jouer sur les code et non sur les index initiaux (n° de ligne)
dfBoxPlot(dfContinues[["proteins_100g", "fat_100g", "carbohydrates_100g"]], "proteins_100g, fat_100g et carbohydrates_100g")
print("Moyenne d'avant annulation de la valeur aberrante : {0}".format(dfContinues[["proteins_100g", "fat_100g", "carbohydrates_100g"]].mean()))
dfContinues = pd.concat([dfContinues, dfOriginel[['code']]], axis = 1)
dfContinues = dfContinues.set_index('code', drop=False)
dfContinues['proteins_100g'][15666666666] = 0
dfContinues['fat_100g'][15666666666] = 0
dfContinues['carbohydrates_100g'][15666666666] = 0  
dfBoxPlot(dfContinues[["proteins_100g", "fat_100g", "carbohydrates_100g"]], "proteins_100g, fat_100g et carbohydrates_100g")
print("Moyenne d'apres annulation de la valeur aberrante : {0}".format(dfContinues[["proteins_100g", "fat_100g", "carbohydrates_100g"]].mean()))


# <h5>Asymétrie des variables</h5>

# Il peut être intéressant de voir les asymétries des variables continues car dans le cas d'un modèle de régression linéaire, on souhaite se rapprocher d'une loi normale, donc diminuer si possible la valeur d'asymétrie (skewness) qui lorsqu'elle est trop forte, pose des problèmes.

# In[143]:

# on supprime la colonne code de dfContinues qui a été rajoutée comme index
dfContinues.drop('code', axis=1, inplace=True)
tabSkewness = []
co = dfContinues.count()
X = []
Y = []
i = 0
while i < dfContinues.shape[1]:    
    if ~np.isnan(dfContinues[dfContinues.columns[i]].skew()):
        tabSkewness.append((dfContinues.columns[i], dfContinues[dfContinues.columns[i]].skew(), dfNanContinues[dfContinues.columns[i]], co[i]))
        Y.append(dfContinues[dfContinues.columns[i]].skew())
        X.append(co[i])
    i = i + 1
print(dfContinues.shape[1])
print(len(tabSkewness))


# <u>Commentaires</u> : <br>
# On observe une diminution du nombre de variables dans le tableau d'asymétrie car pour 6 variables, il n'y a qu'une seule donnée, ce qui est insuffisant pour obtenir une valeur d'asymétrie.

# In[144]:

plt.scatter(X,Y)
plt.xlabel("nombre d'enregistrements")
plt.ylabel("skewness")
plt.show()


# <u>Commentaires</u> : <br>
# A première vue, on ne peut pas tirer de conclusion probante sur le rapport entre le nombre d'informations renseignées et l'asymétrie (si éventuellement le fait d'avoir un très fort ou un très faible nombre d'enregistrements avait eu une influence importante sur l'asymétrie).

# Tableau ordonné des valeurs d'asymétrie :

# In[145]:

tabSkewnessSorted = sorted(tabSkewness, key=lambda x:x[1])
for i in (range(10) + range(len(tabSkewnessSorted)-10, len(tabSkewnessSorted))):
    print(i, tabSkewnessSorted[i][0], tabSkewnessSorted[i][1], tabSkewnessSorted[i][3])

dfContinuesSkewed = pd.DataFrame()
i = 0
while i < len(tabSkewnessSorted):
    dfContinuesSkewed.insert(dfContinuesSkewed.shape[1], tabSkewnessSorted[i][0], dfOriginel[tabSkewnessSorted[i][0]])
    i = i + 1
dfContinuesSkewed['proteins_100g'][15666666666] = 0
dfContinuesSkewed['fat_100g'][15666666666] = 0
dfContinuesSkewed['carbohydrates_100g'][15666666666] = 0  


# <h5>Graphes des distributions des variables continues non vides ayant une valeur de skew</h5>

# In[146]:

dfContinuesSkewed.hist(figsize=(14,50), color="blue", bins=40, normed=True, layout=(17,5))
plt.show()


# <h5>Etude de l'influence d'une transformation sur l'asymétrie des variables</h5>

# Si l'on souhaite diminuer les valeurs d'asymétrie de certaines variables, on peut regarder l'évolution de ces valeurs lors de différentes transformations, par exemple en utilisant les fonctions logarithme et racine carrée

# In[147]:

tabRatLog = []
tabRatRac = []

for i in range(dfContinuesSkewed.shape[1]):
    skewNorm = dfContinuesSkewed[dfContinuesSkewed.columns[i]].skew()   # skew variable originelle X
    logcol = np.log(1+dfContinuesSkewed[dfContinuesSkewed.columns[i]])  # variable log(1 + X)
    skewLog = logcol.skew()                                             # skew variable log(1 + X)
    sqrtcol = np.sqrt(dfContinuesSkewed[dfContinuesSkewed.columns[i]])  # variable sqrt(X)
    skewRac = sqrtcol.skew()                                            # skew variable sqrt(X)
    ratLog = np.abs(skewLog/skewNorm)                                   # ratio skewLog/skewNorm
    tabRatLog.append(np.log(ratLog))
    ratRac = np.abs(skewRac/skewNorm)                                   # ratio skewRac/skewNorm
    tabRatRac.append(np.log(ratRac))


# Graphe du ratio skewLog/skewNorm afin de voir quelles sont les variables dont le passage au log est bénéfique pour le skew :

# In[148]:

plt.figure(1)
x = np.arange(dfContinuesSkewed.shape[1])
y = tabRatLog
plt.plot(x,y,'g')
plt.plot([0, dfContinuesSkewed.shape[1]-1], [0,0], 'r-', lw=2)
plt.xlabel('indices des variables')
plt.ylabel('log(skew Log(V)/skew (V))')
plt.title('Ratio entre la skewness de log(V) et la skewness de V pour une variable V')
plt.show()


# Graphe du ratio skewRac/skewNorm afin de voir quelles sont les variables dont le passage à la racine carrée est bénéfique pour le skew :

# In[149]:

plt.figure(2)
x = np.arange(dfContinuesSkewed.shape[1])
y = tabRatRac
plt.plot(x,y,'b')
plt.plot([0, dfContinuesSkewed.shape[1]-1], [0,0], 'r-', lw=2)
plt.xlabel('indices des variables')
plt.ylabel('log(skew sqrt(V)/skew (V))')
plt.title('Ratio entre la skewness de sqrt(V) et la skewness de V pour une variable V')
plt.show()


# <u>Commentaires</u> : <br>
# On voit donc qu'à partir d'un certain niveau d'asymétrie, utiliser une fonction comme le logarithme ou la racine carrée améliore la valeur de skewness. 

# Par exemple, pour la variable energy_100g : 

# In[150]:

n = tabRatRac.index(min(tabRatRac))
print("On a un log du ratio skew(rac) / skew(norm) de {0}".format(tabRatRac[n]))
skewNorm = dfContinuesSkewed[dfContinuesSkewed.columns[n]].skew()
print("On a en effet une valeur initiale de skew de {0}".format(skewNorm))
sqrtcol = np.sqrt(dfContinuesSkewed[dfContinuesSkewed.columns[n]])
skewRac = sqrtcol.skew()  
print("et une valeur de skew de la racine carrée de la variable de {0}".format(skewRac))
ratRac = np.abs(skewRac/skewNorm)
print("On obtient donc un ratio de {0} et un log du ratio de {1}".format(ratRac, np.log(ratRac)))
print("\nLa transformation de la variable energy_100g à l'aide de la fonction racine carrée permet de diviser son coefficient \nd'asymétrie par plus de 220 ( = {0} / {1})".format(skewNorm, skewRac))  


# On peut essayer de voir s'il n'y a pas d'autres valeurs aberrantes grâce aux boxplots et à un éventuel trop grand écart entre le max et la moyenne ou les 75% : 

# In[151]:

for i in range(dfContinuesSkewed.shape[1]):
    plt.clf()
    sse = dfContinuesSkewed[[dfContinuesSkewed.columns[i]]]
    sse.boxplot()
    print(sse.describe())
    plt.show()


# On voit qu'il y a des cas où une valeur semble excentrée, par exemple pour le molybdenum_100g, la valeur maximum est de 0.003760, ce qui est bien supérieur à la moyenne de 0.000401 (qu'elle participe à bien augmenter) et à la valeur de 0.000075 dont 75% des enregistrements se trouvent en dessous. Nous verrons dans la partie exploration que cet enregistrement de code 3401528535864 peut être repéré dans une ACP. Nous verrons aussi que contrairement à l'enregistrement de code 15666666666, nous n'annulerons pas ses valeurs extrêmes. Cet enregistrement a des valeurs très fortes pour quelques autres variables comme le chromium_100g, le beta-carotene_100g et le copper_100g. Il y a d'autres enregistrements qui possèdent des valeurs très fortes, mais parce qu'ils correspondent à des articles très spécifiques dont ces valeurs fortes sont la spécificité. Ils ne sont pas de nature à bouleverser le jeu de données. 

# Il y a toutefois d'autres enregistrements problématiques qui mériteraient des investigations supplémentaires, comme par exemple une energy_100g à 87217, alors qu'une barre énergisante n'atteint pas les 2000. Ou une eau minérale pour enfant ayant un taux de sel 66 fois plus élevé qu'une sauce soja...
