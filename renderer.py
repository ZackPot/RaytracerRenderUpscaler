# This is AI generated code to create the dataset. All the rest is human-coded, though.

import moderngl
import numpy as np
from PIL import Image
import os
import shutil
import random


# ==============================================================================
# TECHNICAL ATTRIBUTIONS & SPECIFICATIONS
# ------------------------------------------------------------------------------
# 1. Mesh Source: Blender Suzanne Monkey (Standard 3D Test Model)
#    Ref: https://wikipedia.org
#
# 2. Framework: ModernGL (Pythonic OpenGL Wrapper)
#    Ref: https://readthedocs.io
#
# 3. Off-screen Rendering: OpenGL Framebuffer Object (FBO) Logic
#    Ref: https://learnopengl.com
#
# 4. Lighting Model: Three-Point Lighting (Key, Fill, Back)
#    Ref: https://wikipedia.org
# ==============================================================================

class SuzanneDataset:
    def __init__(self, obj_file="suzanne.obj"):
        if os.path.exists("images"):
            shutil.rmtree("images")
        os.makedirs("images")

        self.ctx = moderngl.create_context(standalone=True)
        self.low_res, self.high_res = (100, 100), (2048, 2048)

        vertices, normals = [], []
        with open(obj_file, 'r') as f:
            v_pool, vn_pool = [], []
            for line in f:
                if line.startswith('v '): v_pool.append([float(x) for x in line.split()[1:]])
                if line.startswith('vn '): vn_pool.append([float(x) for x in line.split()[1:]])
                if line.startswith('f '):
                    for part in line.split()[1:]:
                        v_idx, _, vn_idx = [int(i) - 1 for i in part.split('/')]
                        vertices.extend(v_pool[v_idx])
                        normals.extend(vn_pool[vn_idx])

        self.vertex_data = np.array(vertices, dtype='f4')
        self.normal_data = np.array(normals, dtype='f4')

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_vert; in vec3 in_norm;
                out vec3 v_norm;
                uniform mat4 mvp;
                void main() {
                    v_norm = in_norm;
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_norm; out vec4 f_color;
                uniform vec3 obj_color;
                void main() {
                    vec3 n = normalize(v_norm);

                    // Three-point lighting setup for consistent visibility
                    vec3 key_light = normalize(vec3(1.0, 1.0, 1.0));   // Front-Right
                    vec3 fill_light = normalize(vec3(-1.0, 0.0, 1.0)); // Front-Left
                    vec3 back_light = normalize(vec3(0.0, 0.5, -1.0)); // Top-Back

                    float diff = max(dot(n, key_light), 0.0) * 0.8;
                    diff += max(dot(n, fill_light), 0.0) * 0.3;
                    diff += max(dot(n, back_light), 0.0) * 0.2;
                    diff = max(diff, 0.1); // Ambient floor

                    f_color = vec4(obj_color * diff, 1.0);
                }
            """
        )

        self.vbo_v = self.ctx.buffer(self.vertex_data)
        self.vbo_n = self.ctx.buffer(self.normal_data)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo_v, '3f', 'in_vert'), (self.vbo_n, '3f', 'in_norm')])

    def run(self, iterations=1000):
        for i in range(iterations):
            # Keep saturation high for better AI feature detection
            color = (random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0))

            angle = i * (6.283 / iterations)  # Full circle
            rot_y = np.array([
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1.4]  # 1.4 scale to fit in frame
            ], dtype='f4')

            self.prog['mvp'].write(rot_y)
            self.prog['obj_color'].value = color

            self.render_and_save(self.low_res, f"images/lq_{i:04d}.png")
            self.render_and_save(self.high_res, f"images/hq_{i:04d}.png")
            if i % 100 == 0: print(f"Progress: {i}/{iterations}")

    def render_and_save(self, res, path):
        fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(res)],
            depth_attachment=self.ctx.depth_renderbuffer(res)
        )
        fbo.use()
        self.ctx.clear(0.05, 0.05, 0.07)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render()
        Image.frombytes('RGB', res, fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM).save(path)


if __name__ == "__main__":
    SuzanneDataset().run()
