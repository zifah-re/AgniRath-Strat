import requests

URL="http://127.0.0.1:8000/api/"

def get_current_state()->dict:
    req=requests.get(URL+"data/historical")
    data=req.json()
    result={
        "Speed": data['metric']['Speed'],
        "SoC": data['metric']['SOC_Ah'],
        "Distance": data['metric']['distance_travelled']
    }
    return result

def get_profile(profile_list:list)->dict:
    '''
    List of profiles that can be requested ["Altitude","Gradient","Coordinates","Distance","SpeedLimit","SpeedProfile","Headings","TargetProfile"]
    
    Returns a dictionary of profile: profile_data
    '''
    req=requests.get(URL+"data/profile")
    data=req.json()
    result={}
    for profile in profile_list:
        if profile in data['profile'].keys():
            result[profile]=data['profile'][profile]
    return result
    