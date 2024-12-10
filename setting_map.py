import carla
import random
import json

# CARLAサーバーに接続
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# ワールドをロード
town = 'Town02'
world = client.load_world(town)
map = world.get_map()

# トポロジーデータを取得
waypoint_tuple_list = map.get_topology()

# ランダムにスタート地点を選ぶ
way_1st = random.choice(waypoint_tuple_list)

# ゴール地点候補をフィルタリング
way_2nd_candidates = [way for way in waypoint_tuple_list if way[0] == way_1st[1]]
if not way_2nd_candidates:
    raise ValueError("続くゴール地点候補が見つかりません。")

# ランダムにゴール地点を選ぶ
way_2nd = random.choice(way_2nd_candidates)

# コースを表示
print(f"スタート地点: {way_1st[0].transform.location}")
print(f"ゴール地点: {way_2nd[1].transform.location}")

# コース情報をJSON形式で保存
routes = {
    'town': town,
    'way_list': [way_1st, way_2nd]
}

# JSONファイルに保存
with open('route.json', 'w') as json_file:
    json.dump(routes, json_file, indent=4)

print("ルート情報をJSON形式で保存しました: route.json")

