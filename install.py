# Copyright 2021 Yu-Kai Lin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import subprocess as sp
import sys

from rich import traceback
from rich.console import Console

traceback.install()
console = Console()


def exec_command(command: str,
                 capture_output=False,
                 print_output=True) -> sp.CompletedProcess:
    console.print('[bright_black]$ %s' % command)
    if sys.version_info >= (3, 7):
        proc = sp.run(command, shell=True, capture_output=capture_output)
    elif capture_output:
        proc = sp.run(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

        if print_output:
            print(proc.stdout.decode('UTF-8').strip())
            if proc.stderr:
                console.print('[red]' + proc.stderr.decode('UTF-8'))
    else:
        proc = sp.run(command, shell=True)

    if proc.returncode == 0:
        console.print('[bright_black](command terminated with return code = '
                      '%d)' % proc.returncode)
    else:
        console.print('[bold red](command terminated with return code = '
                      '%d)' % proc.returncode)
        raise AssertionError('Subprocess executed with returncode != 0')

    return proc


def get_command_with_augmented_path(command: str,
                                    args: argparse.Namespace) -> str:
    return ' '.join(['PATH=%s' % args.PATH, command])


def check_cmake_version(args: argparse.Namespace) -> None:
    """Check if cmake version is suitable.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments coming from main function.
    """
    console.rule('Checking CMake Version')

    command = 'cmake --version'

    # check default cmake (only for reference)
    console.print('[bold bright_yellow]CMake without augmented PATH:')
    exec_command(command)

    # check cmake with augmented path
    command = get_command_with_augmented_path(command=command, args=args)

    console.print('[bold bright_yellow]CMake with augmented PATH '
                  '(should >= 3.13.2):')
    proc = exec_command(command, capture_output=True)

    _version = re.match(r'cmake version (?P<version>[0-9]+\.[0-9]+\.[0-9]+)',
                        proc.stdout.decode('UTF-8'))
    version = tuple(map(int, _version['version'].split('.')))

    error_message = 'cmake version %s should >= (3, 13, 2)' % str(version)
    assert version >= (3, 13, 2), error_message

    console.print('[bold green]Checked cmake version successfully.')


def dump_python_environment(args):
    console.rule('Dumping Python Environment')

    which_command = 'which %s' % args.python_executable
    version_command = '%s --version' % args.python_executable

    console.print('[bold bright_yellow]Python without augmented PATH: ')
    path = sp.getoutput(which_command)
    version = sp.getoutput(version_command).split()[1]
    console.print('Path    = %s' % path)
    console.print('Version = %s' % version)

    which_command = get_command_with_augmented_path(which_command, args)
    version_command = get_command_with_augmented_path(version_command, args)

    console.print('[bold bright_yellow]Python with augmented PATH: ')
    path = sp.getoutput(which_command)
    version = sp.getoutput(version_command).split()[1]
    console.print('Path    = %s' % path)
    console.print('Version = %s' % version)


def download_3rd_party_dependencies():
    console.rule('Initializing and Updating Git Submodule')
    exec_command('git submodule update --init --recursive')


def install_se_ssd(args: argparse.Namespace):
    console.rule('Installing SE-SSD and ifp-sample')
    command = [
        'cd det3d/core/iou3d',
        '%s setup.py install' % args.python_executable,
    ]
    exec_command(' && '.join(command))

    command = '%s setup.py build develop' % args.python_executable
    exec_command(command)

    command = [
        'cd third-party/ifp-sample',
        '%s -m pip install -e .' % args.python_executable,
    ]
    exec_command(' && '.join(command))


def install_det3d_dependency(args: argparse.Namespace):
    console.rule('Installing dependencies of Det3D')
    command = [
        'cd third-party/spconv',
        get_command_with_augmented_path('%s setup.py bdist_wheel' %
                                        args.python_executable,
                                        args=args),
        'cd ./dist',
        '%s -m pip install *' % args.python_executable,
    ]
    exec_command(' && '.join(command))

    command = [
        'cd third-party/apex',
        '%s -m pip install '
        '-v --disable-pip-version-check '
        '--no-cache-dir '
        '--global-option="--cpp_ext" '
        '--global-option="--cuda_ext" .' % args.python_executable
    ]
    exec_command(' && '.join(command))


def install_ros_numpy(args: argparse.Namespace):
    console.rule('Installing ros_numpy')
    command = [
        'cd third-party/ros_numpy',
        '%s setup.py install' % args.python_executable
    ]
    exec_command(' && '.join(command))


def download_model_weight():
    console.print(
        '[bold bright_yellow]Remember to download model weight from here:\n'
        'https://drive.google.com/file/d/1M2nP_bGpOy0Eo90xWFoTIUkjhdw30Pjs/view'
    )
    console.print('Please put this file to the project root of SE-SSD-ROS.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmake_executable',
                        default='',
                        type=str,
                        help='Binary executable file to cmake (3.13.2+)')
    parser.add_argument('--python_executable',
                        default='python3',
                        type=str,
                        help='Binary executable file to python')

    args = parser.parse_args()
    args_dict = vars(args)

    cmake_bin_dirname: str = os.path.dirname(
        os.path.abspath(args.cmake_executable))
    args_dict['PATH'] = cmake_bin_dirname + ':' + os.environ['PATH']

    args = argparse.Namespace(**args_dict)

    return args


def main():
    args = parse_args()

    check_cmake_version(args)
    dump_python_environment(args)
    download_3rd_party_dependencies()
    install_se_ssd(args)
    install_det3d_dependency(args)
    install_ros_numpy(args)
    download_model_weight()

    console.print('[bold green]Installed SE-SSD successfully.')


if __name__ == '__main__':
    main()
