import json
from   cv2    import cv2
from   pprint import pprint

class rgb_loader:
    def __init__(self, rap3df_database_path):
        self.rgb_front_dict = {}
        
        database_02_structure_path = rap3df_database_path + 'rap3df_data_02/database.json'
        
        with open(database_02_structure_path, 'r') as json_file:
            data = json.load(json_file)
        
        for key, value in list(data.items())[:-1]:
            self.rgb_front_dict[key] = {}

            self.rgb_front_dict[key]['front']  = rap3df_database_path + value['front'][0]['rgb']
            self.rgb_front_dict[key]['left']   = rap3df_database_path + value['left'][0]['rgb']
            self.rgb_front_dict[key]['right']  = rap3df_database_path + value['right'][0]['rgb']
            self.rgb_front_dict[key]['up']     = rap3df_database_path + value['up'][0]['rgb']
            self.rgb_front_dict[key]['down']   = rap3df_database_path + value['down'][0]['rgb']
            self.rgb_front_dict[key]['burned'] = rap3df_database_path + value['burned'][0]['rgb']

    def get_rgb_front_dict(self):
        return self.rgb_front_dict

    def get_rgb_pictures(self, position):
        rgb_front_pictures = {}

        for key, value in list(self.rgb_front_dict.items()):
            rgb_front_pictures[key] = {}

            rgb_front_pictures[key][position]  = cv2.imread(value[position], cv2.COLOR_BGR2RGB)

        return rgb_front_pictures