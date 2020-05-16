"""
Graphical User Interface for Robot in a Maze HMM and inference algos

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

import argparse
import os
import sys

import itertools
import queue
import threading
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import traceback
from importlib.machinery import SourceFileLoader
import io
import numpy as np
# try:
#     from PIL import Image
#     PIL_AVAILABLE = True
# except:
#     PIL_AVAILABLE = False
PIL_AVAILABLE = False

from robot import *
from utils import normalized

DIR_PROBS = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}

DEFAULT_SIMULATION_LENGTH = 20
OBS_0_TEXT = 'No obs for t=0'


def prod(a: int, b: int):
    return itertools.product(range(a), range(b))


def rgb2color(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def create_colormap(rm, gm, bm, n=256):
    x = np.linspace(0, 1, n)
    rv = np.interp(x, rm[:, 0], rm[:, 1])
    gv = np.interp(x, gm[:, 0], gm[:, 1])
    bv = np.interp(x, bm[:, 0], bm[:, 1])
    cmap = np.column_stack((rv, gv, bv))
    cmap[np.isnan(cmap)] = 0
    cmap[cmap > 1] = 1
    cmap[cmap < 0] = 0
    return cmap


def apply_colormap(cmap, x):
    rgb = cmap[x, :]
    rgb256 = np.interp(rgb, [0, 1], [0, 255])
    rgb256 = rgb256.astype('l')
    return tuple(rgb256)


class Container(object):
    def __init__(self, contents=None):
        self._contents = contents

    @property
    def val(self):
        return self._contents

    @val.setter
    def val(self, contents):
        self._contents = contents


class SolutionViewer(tk.Frame):
    TITLE = 'Robot in a Maze - HMM inference algorithms'

    @staticmethod
    def create():
        top = tk.Tk(className=SolutionViewer.TITLE)
        ui = SolutionViewer(top)
        return top, ui

    def __init__(self, master=None, **kw):
        cnf = {}
        super().__init__(master, cnf, **kw)

        self.maze_cont = Container()
        self.environment = None
        self.state_seq = None
        self.obs_seq = None
        self.solver_filename = None
        self.forward = None
        self.forwardbackward = None
        self.viterbi = None
        self.f_beliefs = None
        self.fb_beliefs = None
        self.v_beliefs = None
        self.beliefs = None
        self.solved_queue = queue.Queue()
        self.start_time = None
        self.desired_start_position = None

        self.zoom_var = tk.IntVar(value=40)
        self.time_slice_var = tk.IntVar(value=0)
        self.showing_var = tk.StringVar(value='forward')
        self.current_state_var = tk.StringVar(value='N/A')
        self.current_observation_var = tk.StringVar(value='N/A')
        self.desired_path_length_var = tk.IntVar(value=DEFAULT_SIMULATION_LENGTH)
        self.desired_start_position_var = tk.StringVar(value='')

        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)

        top = self.winfo_toplevel()
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        top.wm_title(self.TITLE)

        self.create_widgets()

    @property
    def maze(self):
        return self.maze_cont.val

    @maze.setter
    def maze(self, m):
        self.maze_cont.val = m

    # noinspection PyAttributeOutsideInit
    def create_widgets(self):
        self.menu_panel = tk.Frame(self, padx=5, pady=5)
        self.menu_panel.grid(column=0, row=0, sticky=tk.N + tk.S)

        # left menu
        menu_row = 0
#        ttk.Separator(self.menu_panel, orient=tk.HORIZONTAL).grid(
#            column=0, row=7, sticky=tk.N + tk.S + tk.W + tk.E, pady=3)

        self.load_maze_button = tk.Button(
            self.menu_panel, text='Load maze',
            command=self._handle_load_maze)
        self.load_maze_button.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.W + tk.E)
        menu_row += 1

        # Path panel
        self.path_panel = tk.LabelFrame(self.menu_panel,
                text="Path management", padx=5, pady=5)
        self.path_panel.grid(column=0, row=menu_row,
                             columnspan=2, sticky=tk.W + tk.E)

        self.load_path_button = tk.Button(
            self.path_panel, text='Load path',
            command=self._handle_load_path)
        self.load_path_button.grid(column=0, row=0, sticky=tk.W + tk.E)

        self.save_path_button = tk.Button(
            self.path_panel, text='Save path',
            command=self._handle_save_path)
        self.save_path_button.grid(column=1, row=0, sticky=tk.W + tk.E)

        self.gen_path_button = tk.Button(
            self.path_panel, text='Generate new path',
            command=self._handle_gen_path)
        self.gen_path_button.grid(column=0, row=1, sticky=tk.W + tk.E)

        self.update_path_button = tk.Button(
            self.path_panel, text='Update current path',
            command=self._handle_update_path)
        self.update_path_button.grid(column=1, row=1, sticky=tk.W + tk.E)

        self.des_path_length_label = tk.Label(
            self.path_panel, text='Desired path length:')
        self.des_path_length_label.grid(column=0, row=2, sticky=tk.E)

        self.des_path_length_value = tk.Entry(
            self.path_panel, textvariable=self.desired_path_length_var)
        self.des_path_length_value.grid(column=1, row=2, sticky=tk.W + tk.E)

        self.des_path_start_label = tk.Label(
            self.path_panel, text='Desired starting position:')
        self.des_path_start_label.grid(column=0, row=3, sticky=tk.W + tk.E)

        self.des_path_start_value = tk.Label(
            self.path_panel, textvariable=self.desired_start_position_var)
        self.des_path_start_value.grid(column=1, row=3, sticky=tk.W + tk.E)


        self.click_label = tk.Label(
            self.path_panel, text='Click in the maze or use')
        self.click_label.grid(column=0, row=4, sticky=tk.E)

        self.random_start_button = tk.Button(
            self.path_panel, text='random starting position',
            command=self._handle_random_start)
        self.random_start_button.grid(column=1, row=4, sticky=tk.W + tk.E)
        # End of path panel
        menu_row += 1

        self.load_solution_button = tk.Button(
            self.menu_panel, text='Load and run solution',
            command=self._handle_load_solution)
        self.load_solution_button.grid(column=0, row=menu_row,
                                   sticky=tk.W + tk.E)

        self.rerun_solution_button = tk.Button(
            self.menu_panel, text='Rerun solution',
            command=self._handle_reload_solution, state=tk.DISABLED)
        self.rerun_solution_button.grid(column=1, row=menu_row,
                                        sticky=tk.W + tk.E)
        menu_row += 1

        self.forward_radio = tk.Radiobutton(
            self.menu_panel, text='Filtering (forward algorithm)',
            variable=self.showing_var, value='forward',command=self._handle_showing)
        self.forward_radio.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.W)
        menu_row += 1

        self.forwardbackward_radio = tk.Radiobutton(
            self.menu_panel, text='Smoothing (forward backward algorithm)',
            variable=self.showing_var, value='forwardbackward',command=self._handle_showing)
        self.forwardbackward_radio.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.W)
        menu_row += 1

        self.viterbi_radio = tk.Radiobutton(
            self.menu_panel, text='ML states (Viterbi algorithm)',
            variable=self.showing_var, value='viterbi', command=self._handle_showing)
        self.viterbi_radio.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.W)
        menu_row += 1

        # Navigation panel
        self.navigation_panel = tk.Frame(self.menu_panel)
        self.navigation_panel.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.N + tk.S)
        menu_row += 1

        panel_column = 0
        self.time_slice_label = tk.Label(
            self.navigation_panel, text='Time slice:')
        self.time_slice_label.grid(column=panel_column, row=0, sticky=tk.W)
        panel_column += 1

        self.to_0_button = tk.Button(
            self.navigation_panel, text='|<<',
            command=lambda: self._handle_nav_button(-10))
        self.to_0_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.minus5_button = tk.Button(
            self.navigation_panel, text='-5',
            command=lambda: self._handle_nav_button(-5))
        self.minus5_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.minus1_button = tk.Button(
            self.navigation_panel, text='-1',
            command=lambda: self._handle_nav_button(-1))
        self.minus1_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.time_slice_entry = tk.Entry(
            self.navigation_panel, textvariable=self.time_slice_var,
            width=3)
        self.time_slice_entry.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.plus1_button = tk.Button(
            self.navigation_panel, text='+1',
            command=lambda: self._handle_nav_button(1))
        self.plus1_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.plus5_button = tk.Button(
            self.navigation_panel, text='+5',
            command=lambda: self._handle_nav_button(5))
        self.plus5_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        self.to_end_button = tk.Button(
            self.navigation_panel, text='>>|',
            command=lambda: self._handle_nav_button(10))
        self.to_end_button.grid(column=panel_column, row=0, sticky=tk.W + tk.E)
        panel_column += 1
        # End of navigation panel

        self.current_position_label = tk.Label(
            self.menu_panel, text='Current robot position:')
        self.current_position_label.grid(column=0, row=menu_row, sticky=tk.W)
        self.current_position_label_val = tk.Label(
            self.menu_panel, textvariable=self.current_state_var)
        self.current_position_label_val.grid(column=1, row=menu_row, sticky=tk.W + tk.E)
        menu_row += 1

        self.current_observation_label = tk.Label(
            self.menu_panel, text='Current robot obs. (NESW):')
        self.current_observation_label.grid(column=0, row=menu_row, sticky=tk.W)
        self.current_observation_label_val = tk.Label(
            self.menu_panel, textvariable=self.current_observation_var)
        self.current_observation_label_val.grid(column=1, row=menu_row, sticky=tk.W + tk.E)
        menu_row += 1

        self.zoom_scale = tk.Scale(self.menu_panel, orient=tk.HORIZONTAL,
                                   label='Cell size (zoom)', command=self.zoom,
                                   from_=20, to=100, variable=self.zoom_var)
        self.zoom_scale.set(30)
        self.zoom_scale.grid(column=0, row=menu_row,
                                   columnspan=2, sticky=tk.W + tk.E)
        menu_row += 1

        self.eps_button = tk.Button(
            self.menu_panel, text='Save canvas as EPS',
            command=lambda: self._handle_eps_button())
        self.eps_button.grid(column=0, row=menu_row, sticky=tk.W + tk.E)
        if PIL_AVAILABLE:
            self.png_button = tk.Button(
                self.menu_panel, text='Save canvas as PNG',
                command=lambda: self._handle_png_button())
            self.png_button.grid(column=1, row=menu_row, sticky=tk.W + tk.E)
        menu_row += 1

        # maze view panel
        self.maze_view = MazeView(self,
                                  self.maze_cont,
                                  self.zoom_var,
                                  self.set_desired_start_position)
        self.maze_view.grid(column=1, row=0, sticky=tk.N + tk.S + tk.W + tk.E)

        self.maze_view.repaint()

        # status bar
        self.status_bar = tk.Label(
            self, text='Solution file: {}'.format(self.solver_filename), bd=1,
            relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(column=0, row=1, columnspan=3, sticky=tk.W + tk.E)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._update_showing_state()
        self._update_navigation_state()
        self._update_export_buttons_state()

    def set_desired_start_position(self, pos):
        self.desired_start_position = pos
        self.desired_start_position_var.set(str(pos))

    # noinspection PyUnusedLocal
    def zoom(self, *args):
        self.maze_view.repaint()

    def _clear_beliefs(self):
        self.beliefs = None
        self.maze_view.set_cell_values(None)
        self.f_beliefs = None
        self.fb_beliefs = None
        self.v_beliefs = None
        self._update_showing_state()

    def _clear_path(self):
        self.state_seq = None
        self.obs_seq = None
        self.time_slice_var.set(0)
        self._update_navigation_state()

    # noinspection PyUnusedLocal
    def _handle_load_maze(self, *args):
        self._clear_beliefs()
        self._clear_path()
        fn = fd.askopenfilename(defaultextension='.map',
                                filetypes=[('MAP', '*.map')],
                                initialdir='./mazes')
        if len(fn) == 0:
            return
        self.load_maze(fn)

    def load_maze(self, fn):
        m = Maze(fn)
        self.maze = m
        self.robot = Robot(ALL_DIRS, DIR_PROBS)
        self.robot.maze = m
        self.robot.set_random_position()
        self.set_desired_start_position(self.robot.position)
        self._handle_gen_path()
        #self._handle_reload_solution()
        self.maze_view.repaint()

    def _handle_load_path(self):
        fn = fd.askopenfilename(defaultextension='.pth',
                                filetypes=[('Path file', '*.pth')],
                                initialdir='.')
        if len(fn) == 0:
            return
        self.load_path(fn)

    def load_path(self, fpath):
        self._clear_path()
        with open(fpath, 'rt', encoding='utf-8') as f:
            first = f.readline().strip()
            pos = tuple(int(x) for x in first.split())
            self.state_seq = [pos]
            self.obs_seq = [OBS_0_TEXT]
            for line in f:
                pos, obs = line.split('|')
                pos = tuple(int(x) for x in pos.strip().split())
                obs = obs.split()
                self.state_seq.append(pos)
                self.obs_seq.append(obs)
        self._display_current_state_obs()
        self._update_navigation_state()
        self._handle_nav_button(0)   # Repaint robot position

    def _handle_save_path(self):
        fpath = fd.asksaveasfilename(
            defaultextension='.pth',
            filetypes=[('PTH', '*.pth')],
            initialdir='.')
        fdir = os.path.dirname(fpath)
        if not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok=True)
        self.save_path(fpath)

    def save_path(self, fpath):
        with open(fpath, 'wt', encoding='utf-8') as f:
            f.write('{:d} {:d}\n'.format(
                self.state_seq[0][0], self.state_seq[0][1]))
            for pos, obs in zip(self.state_seq[1:], self.obs_seq[1:]):
                f.write('{:d} {:d} | '.format(pos[0], pos[1]))
                f.write(' '.join(obs))
                f.write('\n')

    def _handle_gen_path(self):
        self._clear_beliefs()
        self._clear_path()
        self.robot.position = self.desired_start_position
        init_robot_pos = self.robot.position
        states, observs = self.robot.simulate(
            init_robot_pos, self.desired_path_length_var.get())
        self.state_seq = [init_robot_pos] + states
        self.obs_seq = [OBS_0_TEXT] + observs
        self._display_current_state_obs()
        self._update_navigation_state()
        self._handle_nav_button(0)   # Repaint robot position

    def _handle_update_path(self):
        self._clear_beliefs()
        desired_len = self.desired_path_length_var.get()
        current_len = len(self.state_seq) - 1
        steps_to_generate = desired_len - current_len
        if steps_to_generate < 0:
            self.state_seq = self.state_seq[:desired_len+1]
            self.obs_seq = self.obs_seq[:desired_len+1]
            print('Path shortened to the desired length of', desired_len)
        elif steps_to_generate == 0:
            print('The path has the desired length, nothing done.')
        else:
            last_robot_position = self.state_seq[-1]
            new_st, new_obs = self.robot.simulate(
                last_robot_position, steps_to_generate)
            self.state_seq.extend(new_st)
            self.obs_seq.extend(new_obs)
            print('Path prolonged to the desired length of', desired_len)
        self._display_current_state_obs()
        self._update_navigation_state()
        self._handle_nav_button(0)   # Repaint robot position

    def _handle_random_start(self):
        self.desired_start_position = random.choice(self.robot.get_states())
        self.desired_start_position_var.set(
            str(self.desired_start_position))

    def _display_current_state_obs(self):
        current_time = self.time_slice_var.get()
        try:
            state = self.state_seq[current_time]
            self.current_state_var.set(str(state))
        except Exception:
            self.current_state_var.set(str('N/A'))
        try:
            obs = self.obs_seq[current_time]
            self.current_observation_var.set(str(obs))
        except Exception:
            self.current_observation_var.set(str('N/A'))

    # noinspection PyUnusedLocal
    def _handle_load_solution(self, *args):
        fn = fd.askopenfilename(defaultextension='.py',
                                filetypes=[('Python module', '*.py')],
                                initialdir='.')
        if len(fn) == 0:
            return
        self.load_solution(fn)

    # noinspection PyUnusedLocal
    def _handle_reload_solution(self, *args):
        if self.solver_filename is not None:
            self.load_solution(self.solver_filename)

    def load_solution(self, fn):
        self._clear_beliefs()
        self._update_showing_state()
        fn = os.path.abspath(fn)
        self.solver_filename = fn
        self.status_bar.config(text='Solution file: {}'.format(
            self.solver_filename))
        self.rerun_solution_button.config(state=tk.NORMAL)
        try:
            solution_module = SourceFileLoader('module', fn).load_module()
        except Exception as e:
            mb.showerror(e.__class__.__name__,
                         'An exception occurred during loading the solution '
                         'file. Traceback will be written to the '
                         'standard error output.')
            raise
        try:
            self.forward = solution_module.forward
        except Exception as e:
            self.forward = None
            mb.showerror(e.__class__.__name__,
                         'An exception occurred when trying to use '
                         '"forward" symbol. Traceback will be written to '
                         'the standard error output.')
            raise
        try:
            self.forwardbackward = solution_module.forwardbackward
        except Exception as e:
            self.forwardbackward = None
            mb.showerror(e.__class__.__name__,
                         'An exception occurred when trying to use '
                         '"forwardbackward" symbol. Traceback will be written to '
                         'the standard error output.')
            raise
        try:
            self.viterbi = solution_module.viterbi
        except Exception as e:
            self.viterbi = None
            mb.showerror(e.__class__.__name__,
                         'An exception occurred when trying to use '
                         '"forward" symbol. Traceback will be written to '
                         'the standard error output.')
            raise
        threading.Thread(target=self.solve).start()
        self.after(100, self._wait_for_solve)

    # noinspection PyProtectedMember
    def solve(self):
        free_positions = self.maze.get_free_positions()
        cell_vals = normalized({pos: 1 for pos in free_positions})

        if self.forward:
            try:
                self.f_beliefs = self.forward(
                    cell_vals, self.obs_seq[1:], self.robot)
            except Exception as e:
                mb.showerror(e.__class__.__name__,
                             'An exception occurred when executing '
                             '"forward()" function. Traceback will be written to '
                             'the standard error output.')
                print(traceback.format_exc(), file=sys.stderr)
            else:
                self.f_beliefs = [cell_vals] + self.f_beliefs
                self.solved_queue.put("forward")

        if self.forwardbackward:
            try:
                self.fb_beliefs = self.forwardbackward(
                    cell_vals, self.obs_seq[1:], self.robot)
            except Exception as e:
                mb.showerror(e.__class__.__name__,
                             'An exception occurred when executing '
                             '"forwardbackward()" function. Traceback will be written to '
                             'the standard error output.')
                print(traceback.format_exc(), file=sys.stderr)
            else:
                self.fb_beliefs = [cell_vals] + self.fb_beliefs
                self.solved_queue.put("forwardbackward")

        if self.viterbi:
            try:
                path, _ = self.viterbi(
                    cell_vals, self.obs_seq[1:], self.robot)
            except Exception as e:
                mb.showerror(e.__class__.__name__,
                             'An exception occurred when executing '
                             '"viterbi()" function. Traceback will be written to '
                             'the standard error output.')
                print(traceback.format_exc(), file=sys.stderr)
            else:
                # Convert Viterbi positions to beliefs
                self.v_beliefs = [cell_vals]
                for pos in path:
                    self.v_beliefs.append(Counter({pos: 1}))
                self.solved_queue.put("viterbi")
        self.solved_queue.put("all")

    def _wait_for_solve(self):
        try:
            solved = self.solved_queue.get_nowait()
            #self.maze_view.solved = solved
            self._update_showing_state()
            self._handle_showing()
            self._handle_nav_button(0)  # Repaint
            self._update_navigation_state()
            self.maze_view.repaint()
            if solved != "all":
                self.after(100, self._wait_for_solve)
        except queue.Empty:
            self.after(100, self._wait_for_solve)

    def _handle_nav_button(self, step):
        t = self.time_slice_var.get()
        if step == -10:
            t = 0
        elif step == 10:
            t = len(self.obs_seq) - 1
        else:
            t += step
            t = max(t, 0)
            t = min(t, len(self.obs_seq) - 1)
        self.time_slice_var.set(t)
        self._display_current_state_obs()
        self.maze_view.robot_pos = self.state_seq[t] if self.state_seq else None
        if self.beliefs:
            self.maze_view.set_cell_values(self.beliefs[self.time_slice_var.get()])
        self.maze_view.repaint()

    def _handle_showing(self):
        if self.showing_var.get() == 'forward':
            self.beliefs = self.f_beliefs
        elif self.showing_var.get() == 'forwardbackward':
            self.beliefs = self.fb_beliefs
        elif self.showing_var.get() == 'viterbi':
            self.beliefs = self.v_beliefs
        else:
            raise RuntimeError('Unexpected value of showing_var (radiobutton state)')
        if self.beliefs:
            self.maze_view.set_cell_values(self.beliefs[self.time_slice_var.get()])
        else:
            self.maze_view.set_cell_values(None)
        self.maze_view.repaint()

    def _update_showing_state(self):
        f_state = tk.NORMAL if self.f_beliefs else tk.DISABLED
        fb_state = tk.NORMAL if self.fb_beliefs else tk.DISABLED
        v_state = tk.NORMAL if self.v_beliefs else tk.DISABLED
        self.forward_radio.config(state=f_state)
        self.forwardbackward_radio.config(state=fb_state)
        self.viterbi_radio.config(state=v_state)

    def _update_navigation_state(self):
        n_state = tk.NORMAL if self.obs_seq else tk.DISABLED
        self.to_0_button.config(state=n_state)
        self.minus5_button.config(state=n_state)
        self.minus1_button.config(state=n_state)
        self.time_slice_entry.config(state=n_state)
        self.plus1_button.config(state=n_state)
        self.plus5_button.config(state=n_state)
        self.to_end_button.config(state=n_state)

    def _update_export_buttons_state(self):
        if self.maze_view.canvas:
            self.eps_button.config(state=tk.NORMAL)
        else:
            self.eps_button.config(state=tk.DISABLED)
        if PIL_AVAILABLE and self.maze_view.canvas:
            self.png_button.config(state=tk.NORMAL)
        else:
            try:
                self.png_button.config(state=tk.DISABLED)
            except:
                pass

    def _handle_eps_button(self):
        fpath = fd.asksaveasfilename(
            defaultextension='.eps',
            filetypes=[('EPS', '*.eps')],
            initialdir='./frames')
        fdir = os.path.dirname(fpath)
        if not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok=True)
        self.maze_view.save_postscript(fpath)

    def _handle_png_button(self):
        fpath = fd.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png')],
            initialdir='./frames')
        fdir = os.path.dirname(fpath)
        if not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok=True)
        self.maze_view.save_png(fpath)


class MazeView(tk.Frame):

    def __init__(self, master, maze_cont, zoom_var,
                 set_desired_start_position, **kw):
        cnf = {}
        super().__init__(master, cnf, **kw)

        self.node_length = 70
        self.wall_width = 8
        self.wall_color = (191, 191, 191)
        self.robot_color = (255, 255, 0)
        self.offset = (5, 5)
        self.reward_label_color = (255, 255, 255)
        self.normal_color = (0, 0, 0)
        self.special_color = (80, 80, 140)
        self.special_padding = 0.15

        self.maze_cont = maze_cont

        self.zoom_var = zoom_var
        self.set_desired_start_position = set_desired_start_position

        self.label_ids = set()
        self.maze_grid_lines_ids = set()
        self.maze_wall_rects_ids = set()
        self.cell_ids = dict()

        self.value_cmap_neg = create_colormap(np.array([[0.0, 130 / 255],
                                                        [1.0, 0 / 255]]),
                                              np.array([[0.0, 0 / 255],
                                                        [1.0, 0 / 255]]),
                                              np.array([[0.0, 0 / 255],
                                                        [1.0, 0 / 255]])
                                              )
        self.value_cmap_pos = create_colormap(np.array([[0.0, 0 / 255],
                                                        [1.0, 0 / 255]]),
                                              np.array([[0.0, 0 / 255],
                                                        [1.0, 180 / 255]]),
                                              np.array([[0.0, 0 / 255],
                                                        [1.0, 0 / 255]])
                                              )

        self.cell_values = None
        self.min_v = 0
        self.max_v = 1
        self.robot_pos = None

        self._dragging = False
        self.canvas = tk.Canvas(self, bg=rgb2color(*self.normal_color))
        self.canvas.bind('<ButtonRelease-1>', func=self.handle_click)
        self.canvas.bind('<ButtonPress-1>', func=self.scroll_start)
        self.canvas.bind('<B1-Motion>', func=self.scroll_move)
        self.hscroll = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.hscroll.config(command=self.canvas.xview)
        self.vscroll = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.vscroll.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hscroll.set,
                           yscrollcommand=self.vscroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

    @property
    def maze(self):
        return self.maze_cont.val

    @maze.setter
    def maze(self, m):
        self.maze_cont.val = m

    def set_cell_values(self, cell_vals):
        self.cell_values = cell_vals
        if cell_vals:
            self.min_v = min(cell_vals.values())
            self.max_v = max(cell_vals.values())

    def handle_click(self, evt: tk.Event):
        if self._dragging:
            self._dragging = False
            return
        if self.maze is None:
            return
        mx = self.canvas.canvasx(evt.x)
        my = self.canvas.canvasy(evt.y)
        self.set_desired_start_position(
            (int(my/self.node_length), int(mx/self.node_length))
        )

    def scroll_start(self, evt: tk.Event):
        self.canvas.scan_mark(evt.x, evt.y)

    def scroll_move(self, evt: tk.Event):
        self._dragging = True
        self.canvas.scan_dragto(evt.x, evt.y, gain=1)

    def repaint(self):
        self.canvas.delete(tk.ALL)
        if self.maze is None:
            return
        self.node_length = self.zoom_var.get()
        self._draw_walls()
        self._draw_maze()
        self._draw_robot()
        x1, y1, x2, y2 = self.canvas.bbox(tk.ALL)
        self.canvas.config(scrollregion=(x1 - self.offset[0],
                                         y1 - self.offset[1],
                                         x2 + self.offset[0],
                                         y2 + self.offset[1]))

    def _draw_text(self, x: int, y: int, c, text: str, place, anchor):
        if place == tk.SW:
            dx, dy = 0, 1
            ex = self.wall_width / 2 + 2
            ey = -self.wall_width / 2 - 2
        elif place == tk.NE:
            dx, dy = 1, 0
            ex = -self.wall_width / 2 - 2
            ey = self.wall_width / 2 + 2
        elif place == tk.NW:
            dx = dy = 0
            ex = self.wall_width / 2 + 2
            ey = self.wall_width / 2 + 2
        elif place == tk.CENTER:
            dx = dy = .5
            ex = ey = 0
        else:
            raise ValueError('Unsupported text placement.')
        id_ = self.canvas.create_text(
            (x + dx) * self.node_length + ex,
            (y + dy) * self.node_length + ey,
            text=text,
            fill=c,
            anchor=anchor
        )
        self.label_ids.add(id_)

    def _draw_rewards(self):
        for x, y in prod(self.maze.get_width(), self.maze.get_height()):
            self._draw_text(x, y, rgb2color(*self.reward_label_color),
                            '{:.2f}'.format(self.maze.get_reward(x, y)),
                            tk.CENTER, tk.N)

    def _draw_maze(self):
        self.cell_ids.clear()
        for r, c in self.maze.get_free_positions():
            id_ = self._draw_cell(r, c)
            self.cell_ids[id_] = (r, c)

    def _draw_cell(self, r, c):
        x = c * self.node_length
        y = r * self.node_length
        pad = self.special_padding * (self.node_length + self.wall_width)
        f = self._get_cell_color(r, c)
        id_ = self.canvas.create_rectangle(x, y,
                                           x + self.node_length,
                                           y + self.node_length,
                                           width=1,
                                           fill=rgb2color(*f),
                                           outline=rgb2color(*self.wall_color))
        return id_

    def _get_cell_color(self, r, c):
        if not self.cell_values: return self.normal_color
        val = self.cell_values[(r, c)]
        f = self.normal_color
        if val < 0:
            norm_val = int(np.interp(val,
                                     [self.min_v, 0],
                                     [0, 255]))
            value_color = apply_colormap(self.value_cmap_neg, norm_val)
        else:
            norm_val = int(np.interp(val,
                                     [0, self.max_v],
                                     [0, 255]))
            value_color = apply_colormap(self.value_cmap_pos, norm_val)
        return value_color

    def _draw_walls(self):
        h = self.maze.height
        w = self.maze.width

        for iy, ix in self.maze.get_wall_positions():
            x = ix * self.node_length
            y = iy * self.node_length
            id_ = self.canvas.create_rectangle(x, y,
                                               x + self.node_length,
                                               y + self.node_length,
                                               width=1, fill=rgb2color(*self.wall_color),
                                               outline=rgb2color(*self.wall_color))
            self.maze_wall_rects_ids.add(id_)

    def _draw_robot(self):
        if not self.robot_pos: return
        r, c = self.robot_pos
        x = (c+0.25) * self.node_length
        y = (r+0.25) * self.node_length
        id_ = self.canvas.create_oval(x, y,
                                           x + 0.5*self.node_length,
                                           y + 0.5*self.node_length,
                                           width=1,
                                           fill=rgb2color(*self.robot_color),
                                           outline=rgb2color(*self.wall_color))
        return id_

    def _draw_value_labels(self):
        pass

    def _draw_actions(self):
        pass

    def save_postscript(self, fname):
        """Writes the current maze canvas to a postscript file."""
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(self.get_canvas_postscript())

    def get_canvas_postscript(self):
        return self.canvas.postscript(
                pageanchor='sw', colormode='color',
                pagex=0, pagey=0,
                y='0.c', x='0.c')

    def save_png(self, fname):
        ps_code = self.get_canvas_postscript()
        im = Image.open(io.BytesIO(ps_code.encode('utf-8')))
        im.save(fname) #+ '.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MDP Testbed')
    ap.add_argument('-m', '--maze', action='store', required=False, nargs=1,
                    metavar='filename', default=None,
                    help='If specified, the solution viewer or the editor '
                         'will be started with the specified maze already '
                         'loaded. Otherwise no maze will be loaded (and can '
                         'be loaded using the GUI).')
    ap.add_argument('-s', '--solution', action='store', required=False,
                     nargs=1, metavar='filename', default=None,
                     help='If specified, the solution viewer will be launched '
                          'and will immediately load the  solution given in '
                          'the filename. Otherwise no solution will be loaded '
                          '(and can be loaded using the GUI).'
                          'Solution is a Python file containing HMM inference '
                          'functions forward(), forwardbackward(), and viterbi().')
    ns = ap.parse_args()

    top, gui = SolutionViewer.create()
    if ns.maze is not None:
        gui.load_maze(ns.maze[0])
    if ns.solution is not None:
        gui.load_solution(ns.solution[0])
    top.mainloop()
