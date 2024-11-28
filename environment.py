import carla

class Agent:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_control(self, throttle=0.5, steer=0.0):
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        self.vehicle.apply_control(control)
    
    def observe(self):
        # 部分観測（例: センサー情報や周囲の車両位置）
        location = self.vehicle.get_location()
        return {"x": location.x, "y": location.y, "z": location.z}

    def cleanup(self):
        self.vehicle.destroy()

class Hazard:
    def __init__(self, pedestrian):
        self.pedestrian = pedestrian

    def move_randomly(self):
        # 歩行者のランダム移動
        pass

    def cleanup(self):
        self.pedestrian.destroy()