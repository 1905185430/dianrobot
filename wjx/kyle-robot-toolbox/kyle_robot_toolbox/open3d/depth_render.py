'''
深度图渲染器
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class DepthRender:
    '''深度图渲染器'''
    def __init__(self, width=640, height=480, visible=True):
        self.__vis = o3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible)
        self.__width = width
        self.__height = height

        if visible:
            self.poll_events()
            self.update_renderer()

    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = o3d.io.read_pinhole_camera_parameters(filename)
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return image

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return depth

    def capture_depth_image(self, filename):
        self.__vis.capture_depth_image(filename, do_render=True)

        # to read the saved depth image file use:
        # depth = open3d.io.read_image(filename)
        # plt.imshow(depth)
        # plt.show()