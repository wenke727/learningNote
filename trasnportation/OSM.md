# OSM

## [用python小工具下载OSM路网](https://www.cnblogs.com/feffery/p/12483967.html)

我们平时在数据可视化或空间数据分析的过程中经常会需要某个地区的道路网络及节点数据，而OpenStreetMap就是一个很好的数据来源. :airplane:[仓库路径](https://github.com/CNFeffery/DataScienceStudyNotes/blob/master/%EF%BC%88%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%AD%A6%E4%B9%A0%E6%89%8B%E6%9C%AD80%EF%BC%89%E7%94%A8Python%E7%BC%96%E5%86%99%E5%B0%8F%E5%B7%A5%E5%85%B7%E4%B8%8B%E8%BD%BDOSM%E8%B7%AF%E7%BD%91%E6%95%B0%E6%8D%AE/OsmDownloader.py)

``` python
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString
import os
import pandas as pd
from tqdm import tqdm


class OsmDownloader:

    def __init__(self, area):
        self.area = area

    def __search_for_osm_id__(self):
        '''这个函数用于通过nominatim.openstreetmap.org的api来检索目标城市的osm_id'''
        html = requests.get(f'https://nominatim.openstreetmap.org/search?format=json&q={self.area}')

        return [item for item in eval(html.text) if item['type'] == 'administrative']

    def __download__(self):
        self.search_result = self.__search_for_osm_id__()
        self.area_id = int(self.search_result[0]['osm_id'] + 36e8)

        url_front = 'https://overpass.kumi.systems/api/interpreter'
        # TODO
        url_parameters = {'data': f'[timeout:900][maxsize:1073741824][out:json];area({self.area_id});(._; )->.area;(way[highway](area.area); node(w););out skel;'}

        source = requests.post(url_front, data=url_parameters)
        raw_osm_json = eval(source.text)

        # 抽取点图层并保存
        points_contents = []
        for element in tqdm(raw_osm_json['elements'], desc=f'[{self.area}]抽取点数据'):
            if element['type'] == 'node':
                points_contents.append((str(element['id']), element['lon'], element['lat']))

        self.points = pd.DataFrame(points_contents,
                              columns=['id', 'lng', 'lat'])

        self.points['geometry'] = self.points.apply(lambda row: Point([row['lng'], row['lat']]), axis=1)

        self.points = gpd.GeoDataFrame(self.points, crs='EPSG:4326')

        # 构造{id -> 点数据}字典
        self.id2points = {key: value for key, value in zip(self.points['id'],
                                                           self.points['geometry'])}

        # 保存线图层
        ways_contents = []
        for element in tqdm(raw_osm_json['elements'], desc=f'[{self.area}]抽取线数据'):
            if element['type'] == 'way':
                if element['nodes'].__len__() >= 2:
                    ways_contents.append((str(element['id']), LineString([self.id2points[str(_)]
                                                                          for _ in element['nodes']])))

        self.ways = gpd.GeoDataFrame(pd.DataFrame(ways_contents, columns=['id', 'geometry']),
                                     crs='EPSG:4326')

    def download_shapefile(self, path=''):

        try:
            self.search_result
        except AttributeError:
            print('=' * 200)
            print('开始下载数据！')
            self.__download__()
            print('=' * 200)
            print('数据下载完成！')

        print('=' * 200)
        print('开始导出数据！')
        self.points.to_file(os.path.join(path, f'{self.area}_osm路网'), layer='节点')
        self.ways.to_file(os.path.join(path, f'{self.area}_osm路网'), layer='道路')
        print('=' * 200)
        print('数据导出成功！')

    def download_geojson(self, path=''):

        try:
            self.search_result
        except AttributeError:
            print('=' * 200)
            print('开始下载数据！')
            self.__download__()
            print('=' * 200)
            print('数据下载完成！')

        print('=' * 200)
        print('开始导出数据！')
        self.points.to_file(os.path.join(path, f'{self.area}_osm路网_节点.json'), driver='GeoJSON')
        self.ways.to_file(os.path.join(path, f'{self.area}_osm路网_道路.json'), driver='GeoJSON')
        print('=' * 200)
        print('数据导出成功！')
```

