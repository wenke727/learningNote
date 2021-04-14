# 234 · 网页爬虫
# https://www.lintcode.com/problem/234/
from algorithm.Senior_06_DP import TrieNode
import threading
from collections import deque
import re
"""
class HtmlHelper:
    # @param (string)
    # @return (list)
    @classmethod
    def parseUrls(url):
        # Get all urls from a webpage of given url. 
"""

class Solution:
    pattern = re.compile(r"^https?://[^.]*.wikipedia.org")
    pool_size = 3
    pool = set()
    seen = set()
    tasks = deque([])
    results = []

    def crawler(self, url):
        """
        @param url(string): a url of root page
        @return (list): all urls
        """
        self.tasks.append(url)
        self.seen.add(hash(url))
        while len(self.tasks) > 0:
            cur_url = self.tasks.popleft()
            if self.pattern.search(cur_url):
                self.results.append(cur_url)
                for next_url in HtmlHelper.parseUrls(cur_url):
                    t = threading.Thread(target=self._add_task, args=(next_url,))
                    t.start()
                    while True:
                        curts = threading.enumerate()
                        if (len(curts) < self.pool_size):
                            break
        
        return self.results
    
    def _add_task(self, url):
        docid = hash(url)
        if (docid not in self.seen):
            self.tasks.append(url)
            self.seen.add(docid)


# 迷你优步 · Mini Uber
# https://www.lintcode.com/problem/525/
'''
Definition of Trip:
class Trip:
    self.id; # trip's id, primary key
    self.driver_id, self.rider_id; # foreign key
    self.lat, self.lng; # pick up location
    def __init__(self, rider_id, lat, lng):

Definition of Helper
class Helper:
    @classmethod
    def get_distance(cls, lat1, lng1, lat2, lng2):
        # return calculate the distance between (lat1, lng1) and (lat2, lng2)
'''
from Trip import Trip, Helper

class Location:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

class MiniUber:
    def __init__(self):
        # initialize your data structure here.
        self.driver2Location = {}
        self.drver2Trip = {}

    def report(self, driver_id, lat, lng):
        # @param {int} driver_id an integer
        # @param {double} lat, lng driver's location
        # return {trip} matched trip information if there have matched rider or null
        if driver_id in self.drver2Trip:
            return self.drver2Trip[driver_id]
        
        if driver_id in self.driver2Location:
            self.driver2Location[driver_id].lat = lat
            self.driver2Location[driver_id].lng = lng
        else:
            self.driver2Location[driver_id] = Location(lat, lng)
        
        return None

    def request(self, rider_id, lat, lng):
        # @param rider_id an integer
        # @param lat, lng rider's location
        # return a trip
        trip = Trip(rider_id, lat, lng)
        distance, driver_id = -1, -1

        for key, val in self.driver2Location.items():
            dis = Helper.get_distance( val.lat, val.lon, lat, lng)
            if distance < 0 or distance > dis:
                driver_id = key
                distance = dis

        if driver_id != -1:
            del self.driver2Location[driver_id]
        
        trip.driver_id = driver_id
        self.drver2Trip[driver_id] = trip

        return trip



# 560 · 友谊服务
# https://www.lintcode.com/problem/560/
class FriendshipService:
    def __init__(self):
        self.followers = {}
        self.followings = {}

    def getFollowers(self, user_id):
        if user_id not in self.followers: 
            return []
        
        res = list( self.followers[user_id] )
        res.sort()

        return res

    def getFollowings(self, user_id):
        if user_id not in self.followings:
            return []
        results = list(self.followings[user_id])
        results.sort()
        return results        

    def follow(self, to_user_id, from_user_id):
        if to_user_id not in self.followers:
            self.followers[to_user_id] = set()
        self.followers[to_user_id].add(from_user_id)

        if from_user_id not in self.followings:
            self.followings[from_user_id] = set()
        self.followings[from_user_id].add(to_user_id)

    def unfollow(self, to_user_id, from_user_id):
        if to_user_id in self.followers:
            if from_user_id in self.followers[to_user_id]:
                self.followers[to_user_id].remove(from_user_id)

        if from_user_id in self.followings:
            if to_user_id in self.followings[from_user_id]:
                self.followings[from_user_id].remove(to_user_id)



import re
class HtmlParser:
    def parseUrls(self, content):
        # \s是指空白，包括空格、换行、tab缩进等所有的空白，而\S刚好相反
        # * 号代表前面的字符可以不出现，也可以出现一次或者多次（0次、或1次、或多次）
        links = re.findall(r'\s*(?i)href\s*=\s*("|\')+([^"\'>\s]*)', content, re.I)
        return [link[1] for link in links if len(link[1]) and not link[1].startswith('#')]


# 502 · 迷你Cassandra
# https://www.lintcode.com/problem/502/
from collections import OrderedDict
class MiniCassandra:
    def __init__(self):
        self.hash = {}

    def insert(self, row_key, column_key, column_value):
        if row_key not in self.hash:
            self.hash[row_key] = OrderedDict()
        
        self.hash[row_key][column_key] = column_value
    
        return 
    
    def query(self, row_key, column_start, column_end):
        rt = []
        if row_key not in self.hash:
            return rt
        
        self.hash[row_key] = OrderedDict(sorted(self.hash[row_key].items()))
        for key, val in self.hash[row_key].items():
            if column_start <= key <= column_end:
                rt.append( Column(key, val) )
        
        return rt

# 509 · 迷你Yelp
# https://www.lintcode.com/problem/509/
'''
Definition of Location:
class Location:
    # @param {double} latitude, longitude
    # @param {Location}
    @classmethod
    def create(cls, latitude, longitude):
        # This will create a new location object

Definition of Restaurant:
class Restaurant:
    # @param {str} name
    # @param {Location} location
    # @return {Restaurant}
    @classmethod
    def create(cls, name, location):
        # This will create a new restaurant object,
        # and auto fill id

Definition of Helper
class Helper:
    # @param {Location} location1, location2
    @classmethod
    def get_distance(cls, location1, location2):
        # return calculate the distance between two location

Definition of GeoHash
class GeoHash:
    # @param {Location} location
    # @return a string
    @classmethom
    def encode(cls, location):
        # return convert location to a geohash string
    
    # @param {str} hashcode
    # @return {Location}
    @classmethod
    def decode(cls, hashcode):
        # return convert a geohash string to location
'''
from YelpHelper import Location, Restaurant, GeoHash, Helper
from collections import defaultdict
from heapq import heappush, heappop
class MiniYelp:
    def __init__(self):
        self.errors = [2500, 630, 78, 20, 2.4, 0.61, 0.076, 0.01911]
        self.id2restaurant = dict()
        self.loc_table = defaultdict(lambda: defaultdict(set))

    def add_restaurant(self, name, location):
        restaurant = Restaurant.create(name, location)
        r_id = restaurant.id

        self.id2restaurant[r_id] = restaurant
        
        geocode = GeoHash.encode(location)
        
        for i in range(9):
            self.loc_table[i][geocode[:i]].add(r_id)
        
        return r_id

    def remove_restaurant(self, restaurant_id):
        if restaurant_id not in self.id2restaurant:
            return
        
        restaurant = self.id2restaurant.pop(restaurant_id)
        geocode = GeoHash.encode(restaurant.location)
        for i in range(9):
            self.loc_table[i][geocode[:i]].remove(restaurant_id)
        
    def neighbors(self, location, k): 
        code_len = self.get_code_len(k)
        geocode = GeoHash.encode(location)

        r_ids = list(self.loc_table[code_len][geocode[:code_len]])
        dist_lists = []
        for r_id in r_ids:
            restaurant = self.id2restaurant[r_id]
            dist = Helper.get_distance(restaurant.location, location)
            if dist < k:
                heappush(dist_lists, (dist, restaurant.name))
        
        name_lists = []
        while dist_lists:
            _, name = heappop(dist_lists)
            name_lists.append(name)
            
        return name_lists
        
    def get_code_len(self, k):
        for i, error in enumerate(self.errors):
            if k > error:
                return i
        return len(self.errors)


# GFS客户端 · GFS Client
# https://www.jiuzhang.com/solution/gfs-client/
'''
Definition of BaseGFSClient
class BaseGFSClient:
    def readChunk(self, filename, chunkIndex):
        # Read a chunk from GFS
    def writeChunk(self, filename, chunkIndex, content):
        # Write a chunk to GFS
'''
class GFSClient(BaseGFSClient):
    def __init__(self, chunkSize):
        BaseGFSClient.__init__(self)
        self.chunkSize = chunkSize
        self.chunkNum = dict()


    def read(self, filename):
        if filename not in self.chunkNum:
            return None
        
        content = ''
        for index in range(self.chunkNum.get(filename)):
            sub_content = BaseGFSClient.readChunk(self, filename, index)
            if sub_content:
                content += sub_content

        return content

    def write(self, filename, content):
        length = len(content)
        chunkNum = int((length - 1) / self.chunkSize) + 1
        self.chunkNum[filename] = chunkNum

        for index in range(chunkNum):
            sub_content = content[index * self.chunkSize :
                                  (index + 1) * self.chunkSize]
            BaseGFSClient.writeChunk(self, filename, index, sub_content)


# 查找树服务 · Trie Service
# https://www.jiuzhang.com/solution/trie-service/
"""
Definition of TrieNode:
class TrieNode:
    def __init__(self):
        # <key, value>: <Character, TrieNode>
        self.children = collections.OrderedDict()
        self.top10 = []
"""
class TrieService:
    def __init__(self):
        self.root = TrieNode()
    
    def get_root(self):
        return self.root
    
    def insert(self, word, frequency):
        node = self.root

        for letter in word:
            node.children[letter] = node.children.get(letter, TrieNode())
            self.add_frenquency(node.children[letter].top10, frequency)

            node = node.children[letter]

    def add_frequency(self, top10, frequency):
        top10.append(frequency)
        index = len(top10) - 1
        while index > 0:
            if top10[index] > top10[index-1]:
                top10[index], top10[index-1] = top10[index-1], top10[index]
                index -= 1
            else:
                break
        
        if len(top10) > 10: top10.pop()


# 232 · Tiny Url
# https://www.lintcode.com/problem/232/
class TinyUrl:
    
    def __init__(self):
        self.dict = {}

    def getShortKey(self, url):
        return url[-6:]

    def idToShortKey(self, id):
        ch = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        s = ""
        while id > 0:
            s = ch[id % 62] + s
            id /= 62
        while len(s) < 6:
            s = 'a' + s
        return s

    def shortkeyToid(self, short_key):
        id = 0
        for c in short_key:
            if 'a' <= c and c <= 'z':
                id = id * 62 + ord(c) - ord('a')
            if 'A' <= c and c <= 'Z':
                id = id * 62 + ord(c) - ord('A') + 26
            if '0' <= c and c <= '9':
                id = id * 62 + ord(c) - ord('0') + 52

        return id

    def longToShort(self, url):
        ans = 0
        for a in url:
            ans = (ans * 256 + ord(a)) % 56800235584L

        while ans in self.dict and self.dict[ans] != url:
            ans = (ans + 1) % 56800235584L

        self.dict[ans] = url
        return "http://tiny.url/" + self.idToShortKey(ans)

    def shortToLong(self, url):
        short_key = self.getShortKey(url)
        return self.dict[self.shortkeyToid(short_key)]


# 565 · 心跳
# https://www.lintcode.com/problem/565/
class HeartBeat:
    def __init__(self):
        self.slaves_ip_list = dict()

    def initialize(self, slaves_ip_list, k):
        self.k = k
        for ip in slaves_ip_list:
            self.slaves_ip_list[ip] = 0

    def ping(self, timesatmp, slave_ip):
        if slave_ip not in self.slaves_ip_list:
            return
        
        self.slaves_ip_list[slave_ip] = timesatmp

    def getDiedSlaves(self, timestamp):
        res = []
        for ip, t in self.slaves_ip_list.items():
            if t <= timestamp - 2*self.k:
                res.append(ip)

        return res


# 538 · 内存缓存
# https://www.lintcode.com/problem/538/
class Resource:
    def __init__(self, value, expired):
        self.value = value
        self.expired = expired

INT_MAX = 0x7fffffff

class Memcache:
    def __init__(self):
        self.client = dict()
    
    def get(self, curtTime, key):
        if key not in self.client:
            return INT_MAX
        res = self.client.get(key)
        if res.expired >= curtTime or res.expired == -1:
            return res.value
        else:
            return INT_MAX

    def set(self, curtTime, key, value, ttl):
        if ttl:
            res = Resource(value, curtTime + ttl - 1)
        else:
            res = Resource(value, -1)

        self.client[key] = res 
        
    def delete(self, curtTime, key):
        if key not in self.client:
            return

        del self.client[key]

    def incr(self, curtTime, key, delta):
        if self.get(curtTime, key) == INT_MAX:
            return INT_MAX
        self.client[key].value += delta

        return self.client[key].value

    def decr(self, curtTime, key, delta):
        if self.get(curtTime, key) == INT_MAX:
            return INT_MAX
        self.client[key].value -= delta

        return self.client[key].value


# 标准型布隆过滤器 · Standard Bloom Filter
# https://www.jiuzhang.com/solution/standard-bloom-filter/
import random
class HashFunction:  
    def __init__(self, cap, seed):
        self.cap = cap
        self.seed = seed
    
    def hash(self, value):
        ret = 0
        for i in value:
            ret += self.seed * ret + ord(i)
            ret %= self.cap

        return ret   

class StandardBloomFilter:
    def __init__(self, k):
        self.bitset = dict()
        self.hashFunc = []
        for i in range(k):
            self.hashFunc.append(HashFunction(random.randint(10000, 20000), i * 2 + 3))
        
    def add(self, word):
        for f in self.hashFunc:
            position = f.hash(word)
            self.bitset[position] = 1
            
    def contains(self, word):
        for f in self.hashFunc:
            position = f.hash(word)
            if position not in self.bitset:
                return False
       
        return True



