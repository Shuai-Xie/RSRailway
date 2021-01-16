import requests
import random
import json

"""
利用高德地图api实现经纬度与地址的批量转换
"""

key = '24823293136d6fd8244d1140e6cc6c04'


# 坐标转换是一类简单的HTTP接口，能够将用户输入的非高德坐标（GPS坐标、mapbar坐标、baidu坐标）转换成高德坐标
def transform(location):
    # coordsys：gps;mapbar;baidu;autonavi(不进行转换)
    parameters = {
        'coordsys': 'baidu',
        'locations': location,
        'key': key
    }
    base = 'https://restapi.amap.com/v3/assistant/coordinate/convert?'
    response = requests.get(base, parameters)
    answer = response.json()
    return answer['locations']
    # 我这里是将经纬度转换为地址，所以选用的是逆地理编码的接口。


def geocode(location):
    parameters = {
        'location': location,
        'key': key
    }
    base = 'http://restapi.amap.com/v3/geocode/regeo'
    response = requests.get(base, parameters)
    answer = response.json()
    return answer


def getdetail(detail):
    formatted_address = detail['regeocode']['formatted_address']
    addressComponent = detail['regeocode']['addressComponent']  # 这个 key 是 addressComponent，不能换
    # print('formatted_address', formatted_address)
    # print('addressComponent')
    # pprint.pprint(addressComponent)
    return addressComponent


def get_random_city(allcitys):
    province = random.choice(list(allcitys.keys()))  # 随机选取一个省份
    city = random.choice(list(allcitys[province]))  # 在该省下随机选取一个城市
    return allcitys[province][city]  # list, as [117.17, 31.52]


def get_img_info(location):
    location = ','.join(map(str, location))  # list -> str
    detail = geocode(location)
    return getdetail(detail)


def get_weather_info(city):
    weatherJsonUrl = "http://wthrcdn.etouch.cn/weather_mini?city=" + city  # 将链接定义为一个字符串
    response = requests.get(weatherJsonUrl)  # 获取并下载页面，其内容会保存在respons.text成员变量里面
    response.raise_for_status()  # 这句代码的意思如果请求失败的话就会抛出异常，请求正常就上面也不会做

    # 将json文件格式导入成python的格式
    weatherData = json.loads(response.text)
    weather_dict = dict()
    weather_dict['high'] = weatherData['data']['forecast'][0]['high']
    weather_dict['low'] = weatherData['data']['forecast'][0]['low']
    weather_dict['type'] = weatherData['data']['forecast'][0]['type']
    weather_dict['fengxiang'] = weatherData['data']['forecast'][0]['fengxiang']
    weather_dict['ganmao'] = weatherData['data']['ganmao']
    return weather_dict


if __name__ == '__main__':
    locations = [
        # '116.403847,39.915526',  # ,后面不能有空格
        '120.129371,30.270192',
        # '120.09334,30.308693',
        # '120.219396,30.297149'
    ]
    key = '24823293136d6fd8244d1140e6cc6c04'
    # frombaidu = 1  # 如果坐标点来自百度地图,需要进行坐标变换
    for location in locations:
        # if frombaidu == 1:
        #     location = transform(location)
        detail = geocode(location)
        getdetail(detail)
