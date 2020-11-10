# Análisis estadístico a siniestros con ciclos involucrados


```python
# librerías GENERALES
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import dateutil, datetime # para trabajar con fechas
```


```python
# carpetas de trabajo
carpeta_entrada = '/Users/ignacioabe/personal/recursos/geodatos/vectorial/por-tema/transporte/siniestros CONASET/siniestros-bici/2-procesados/'
carpeta_salida = '/Users/ignacioabe/personal/recursos/geodatos/vectorial/por-tema/transporte/siniestros CONASET/siniestros-bici/3-otros-derivados'
```


```python
# lectura datos
df = pd.read_csv(os.path.join(carpeta_salida, '2013-2019-comunas.csv'))
print(df.shape)
df.head()
```

    (7801, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FID</th>
      <th>año</th>
      <th>x</th>
      <th>y</th>
      <th>fallecidos</th>
      <th>graves</th>
      <th>menos_graves</th>
      <th>leves</th>
      <th>total</th>
      <th>geometry</th>
      <th>index_right</th>
      <th>comuna</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2013</td>
      <td>-70.753096</td>
      <td>-33.695480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>POINT (-70.75309560018178 -33.69548000020183)</td>
      <td>8</td>
      <td>San Bernardo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2013</td>
      <td>-70.704797</td>
      <td>-33.610151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>POINT (-70.70479741976504 -33.61015145842305)</td>
      <td>8</td>
      <td>San Bernardo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2013</td>
      <td>-70.696925</td>
      <td>-33.611317</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>POINT (-70.69692469791642 -33.6113165346303)</td>
      <td>8</td>
      <td>San Bernardo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2013</td>
      <td>-70.695285</td>
      <td>-33.617282</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>POINT (-70.6952853507383 -33.61728230973075)</td>
      <td>8</td>
      <td>San Bernardo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2013</td>
      <td>-70.690355</td>
      <td>-33.620017</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>POINT (-70.69035483111054 -33.62001695022202)</td>
      <td>8</td>
      <td>San Bernardo</td>
    </tr>
  </tbody>
</table>
</div>



## Inicio análisis


```python
# incidentes por año
df['año'].value_counts().sort_index()
```




    2013     960
    2014    1690
    2015     973
    2016    1024
    2017    1178
    2018     989
    2019     987
    Name: año, dtype: int64




```python
# total de heridos y fallecidos por año
df.groupby('año')['total'].sum().sort_index()
```




    año
    2013    1000
    2014    2158
    2015     992
    2016     899
    2017    1176
    2018    1014
    2019     980
    Name: total, dtype: int64




```python
# total de fallecidos por año
df.groupby('año')['fallecidos'].sum().sort_index()
```




    año
    2013     26
    2014    117
    2015     28
    2016     13
    2017     19
    2018     13
    2019     20
    Name: fallecidos, dtype: int64




```python
# total de incidentes por comuna
df['comuna'].value_counts()
```




    Santiago               910
    Providencia            664
    Ñuñoa                  529
    Puente Alto            481
    Maipú                  449
    Las Condes             376
    La Florida             310
    Quinta Normal          291
    Quilicura              267
    Pudahuel               243
    Peñalolén              238
    El Bosque              232
    Macul                  191
    Recoleta               189
    San Joaquín            188
    Estación Central       188
    La Reina               178
    La Cisterna            152
    Cerro Navia            151
    Renca                  146
    La Granja              146
    Lo Prado               131
    San Ramón              128
    Pedro Aguirre Cerda    124
    Vitacura               121
    La Pintana             120
    Independencia           91
    San Bernardo            90
    Conchalí                90
    Huechuraba              87
    Cerrillos               76
    San Miguel              74
    Lo Barnechea            69
    Colina                  25
    Lo Espejo               25
    Lampa                   23
    Padre Hurtado            7
    Pirque                   1
    Name: comuna, dtype: int64




```python
# total de personas fallecidas por comuna
df.groupby('comuna')['fallecidos'].sum().sort_values(ascending=False)
```




    comuna
    Puente Alto            25
    Santiago               13
    La Florida             13
    Estación Central       13
    Quilicura              13
    Maipú                  12
    Quinta Normal          10
    Conchalí                9
    La Pintana              8
    Renca                   8
    El Bosque               7
    Peñalolén               7
    Ñuñoa                   7
    Recoleta                7
    San Joaquín             7
    La Granja               6
    Vitacura                6
    Lo Espejo               6
    Providencia             6
    Pedro Aguirre Cerda     5
    San Miguel              5
    La Cisterna             5
    Cerro Navia             5
    Pudahuel                4
    Las Condes              4
    La Reina                4
    Huechuraba              3
    Macul                   3
    Lo Prado                3
    San Bernardo            3
    Cerrillos               3
    Lampa                   2
    San Ramón               2
    Lo Barnechea            1
    Independencia           1
    Padre Hurtado           0
    Pirque                  0
    Colina                  0
    Name: fallecidos, dtype: int64




```python
# total de personas con lesiones graves por comuna
df.groupby('comuna')['graves'].sum().sort_values(ascending=False)
```




    comuna
    Santiago               189
    Providencia            167
    Ñuñoa                  134
    Puente Alto            103
    Las Condes              84
    Pudahuel                75
    Peñalolén               72
    Quinta Normal           72
    La Florida              71
    Maipú                   63
    San Joaquín             50
    Quilicura               47
    Estación Central        44
    La Pintana              43
    El Bosque               42
    Recoleta                41
    Pedro Aguirre Cerda     39
    La Cisterna             38
    La Reina                37
    Macul                   37
    Vitacura                37
    Cerro Navia             34
    La Granja               32
    Renca                   32
    Lo Prado                30
    Conchalí                27
    Lo Barnechea            25
    Independencia           20
    San Ramón               19
    Cerrillos               14
    San Miguel              13
    San Bernardo            13
    Huechuraba              12
    Lampa                    7
    Lo Espejo                3
    Colina                   1
    Pirque                   0
    Padre Hurtado            0
    Name: graves, dtype: int64




```python
# total de personas con lesiones o fallecias por comuna
df.groupby('comuna')['total'].sum().sort_values(ascending=False)
```




    comuna
    Santiago               918
    Providencia            696
    Ñuñoa                  628
    Puente Alto            524
    Maipú                  466
    Las Condes             401
    La Florida             339
    Quinta Normal          307
    Quilicura              294
    Pudahuel               258
    Peñalolén              255
    El Bosque              240
    Estación Central       199
    Recoleta               196
    La Reina               195
    Macul                  191
    San Joaquín            176
    Cerro Navia            160
    La Cisterna            156
    La Granja              149
    Pedro Aguirre Cerda    145
    La Pintana             144
    Renca                  141
    San Ramón              138
    Lo Prado               129
    Vitacura               122
    Independencia          101
    Conchalí               100
    Huechuraba              80
    Cerrillos               79
    Lo Barnechea            79
    San Bernardo            72
    San Miguel              60
    Lo Espejo               27
    Lampa                   24
    Colina                  23
    Padre Hurtado            6
    Pirque                   1
    Name: total, dtype: int64




```python
# total de personas con lesiones o fallecidas por comuna, año 2019
df[df['año'] == 2019].groupby('comuna')['total'].sum().sort_values(ascending=False)
```




    comuna
    Santiago               140
    Providencia            103
    Maipú                   88
    Ñuñoa                   64
    Puente Alto             61
    La Florida              51
    Quinta Normal           33
    La Reina                33
    Estación Central        31
    San Joaquín             27
    El Bosque               27
    Quilicura               26
    San Ramón               25
    Pudahuel                24
    Recoleta                23
    Lo Prado                20
    Las Condes              20
    San Bernardo            20
    Peñalolén               19
    La Cisterna             18
    Renca                   18
    Cerro Navia             18
    Macul                   14
    Cerrillos               14
    Independencia           10
    San Miguel              10
    Vitacura                 9
    Conchalí                 8
    Huechuraba               7
    La Granja                6
    Pedro Aguirre Cerda      6
    Lo Espejo                3
    La Pintana               2
    Lo Barnechea             2
    Name: total, dtype: int64




```python
# total de personas con lesiones o fallecidas por año, comuna de Santiago
df[df['comuna'] == 'Santiago'].groupby('año')['total'].sum()#.sort_values(ascending=False)
```




    año
    2013    156
    2014    173
    2015    114
    2016    101
    2017    130
    2018    104
    2019    140
    Name: total, dtype: int64




```python
# total de personas con lesiones o fallecidas por año, comuna de Providencia
df[df['comuna'] == 'Providencia'].groupby('año')['total'].sum()#.sort_values(ascending=False)
```




    año
    2013     49
    2014    127
    2015    106
    2016     84
    2017     76
    2018    151
    2019    103
    Name: total, dtype: int64




```python
# tabla con fallecidos y lesionados según gravedad, por año
df.groupby('año')[['fallecidos', 'graves', 'menos_graves', 'leves', 'total']].sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fallecidos</th>
      <th>graves</th>
      <th>menos_graves</th>
      <th>leves</th>
      <th>total</th>
    </tr>
    <tr>
      <th>año</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013</th>
      <td>26</td>
      <td>150</td>
      <td>59</td>
      <td>765</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>117</td>
      <td>574</td>
      <td>167</td>
      <td>1300</td>
      <td>2158</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>28</td>
      <td>164</td>
      <td>54</td>
      <td>746</td>
      <td>992</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>13</td>
      <td>207</td>
      <td>67</td>
      <td>612</td>
      <td>899</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>19</td>
      <td>229</td>
      <td>85</td>
      <td>843</td>
      <td>1176</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>13</td>
      <td>236</td>
      <td>70</td>
      <td>695</td>
      <td>1014</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>20</td>
      <td>207</td>
      <td>67</td>
      <td>686</td>
      <td>980</td>
    </tr>
  </tbody>
</table>
</div>




```python
# totales de personas fallecidas y con lesiones en el período
df[['fallecidos', 'graves', 'menos_graves', 'leves', 'total']].sum()
```




    fallecidos       236
    graves          1767
    menos_graves     569
    leves           5647
    total           8219
    dtype: int64




```python
# accidentes sin daños personales registrados
sel = df[df['total'] == 0]
sel.shape
```




    (1071, 12)




```python
df.shape
```




    (7801, 12)



## Parte interactiva


```python
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
```


```python
def f(x):
    return x
```


```python
interact(f, x=['apples','oranges']);
```


    interactive(children=(Dropdown(description='x', options=('apples', 'oranges'), value='apples'), Output()), _do…



```python
comunas = df['comuna'].unique()
comunas.sort()
print(comunas)
```

    ['Cerrillos' 'Cerro Navia' 'Colina' 'Conchalí' 'El Bosque'
     'Estación Central' 'Huechuraba' 'Independencia' 'La Cisterna'
     'La Florida' 'La Granja' 'La Pintana' 'La Reina' 'Lampa' 'Las Condes'
     'Lo Barnechea' 'Lo Espejo' 'Lo Prado' 'Macul' 'Maipú' 'Padre Hurtado'
     'Pedro Aguirre Cerda' 'Peñalolén' 'Pirque' 'Providencia' 'Pudahuel'
     'Puente Alto' 'Quilicura' 'Quinta Normal' 'Recoleta' 'Renca'
     'San Bernardo' 'San Joaquín' 'San Miguel' 'San Ramón' 'Santiago'
     'Vitacura' 'Ñuñoa']



```python
def anual(comuna_sel):
    return df[df['comuna'] == comuna_sel].groupby('año')['total'].sum()
```


```python
interact(anual, comuna_sel=comunas);
```


    interactive(children=(Dropdown(description='comuna_sel', options=('Cerrillos', 'Cerro Navia', 'Colina', 'Conch…



```python
cat_gravedad = ['fallecidos', 'graves', 'menos_graves', 'leves', 'total']
```


```python
def anual(comuna_sel, gravedad):
    return df[df['comuna'] == comuna_sel].groupby('año')[gravedad].sum()
```


```python
interact(anual, comuna_sel=comunas, gravedad=cat_gravedad);
```


    interactive(children=(Dropdown(description='comuna_sel', options=('Cerrillos', 'Cerro Navia', 'Colina', 'Conch…

