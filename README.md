# Repo for computational tasks.

## Task list:
1. Term 1:
    * `S1T0_types_convertions`
    * `S1T1_compute_function`
    * `S1T2_solve_linear_system`
    * `S1T3_newton_method`
1. Term 2:
    * `S2T1_optimization`
    * `S2T2_interpolation`
    * `S2T3_approximation`
1. Term 3:
    * `S3T1_integration`
    * `S3T2_solve_ode`

Each task should be solved in Python>=3.6, look corresponding folders for code examples and test scripts.
You should write the code that solves the problem and is verified by test script. Check readme.md files in subfolders for details (in Russian!).

## Common Python code structure:
* `<TASK_NAME>/`
    * `__init__.py`
    * `py/`
        * `__init__.py`
        * `test_<subtask_name>.py` - test script that checks your solutions. You have several options to run it:
          * `python -m pytest path/to/test_<task_name>.py`
          * `pytest path/to/test_<task_name>.py`
          * PyCharm: see below
        * `<subtask_name>.py` - dummy code, you should replace it with yours.

## 7 steps to start from scratch
1. Install Anaconda with Python>=3.6: https://www.anaconda.com/products/individual
2. Install PyCharm: https://www.jetbrains.com/ru-ru/pycharm/download
3. Download and unpack https://github.com/sedefe/chm/archive/master.zip (or `git clone` it if you could).
4. In Pycharm open the project in main folder (`chm/` or `chm-master/`).
5. Set the path to Anaconda interpreter: (`File`->`Settings`->`Project`->`Project Interpreter`).
6. Set pytest as a Default test runner (`File`->`Settings`->`Tools`->`Python Integrated Tools`).
7. Choose any `test_*.py` file and press green `Run ...` button at any `def test_*():` row.

## Viewing notebooks online
Github has issues with showing notebooks. You may use https://nbviewer.jupyter.org/ instead and paste the link to `.ipynb` file there.

For example, https://nbviewer.jupyter.org/github/sedefe/chm/blob/master/S2T2_interpolation/notebooks/hermite-interpolation.ipynb

## Contacts
Feel free to ask:
* sagoyewatha@mail.ru
* [Term 1] https://t.me/chm_1sem
* [Term 3] https://t.me/chm_3sem
