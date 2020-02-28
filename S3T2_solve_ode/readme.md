Методичка: http://www.apmath.spbu.ru/ru/staff/eremin/files/task8_2016.pdf

Во вложении тесты для ваших численных решений задачи Коши.

* `py/`
	* `test_fix_step.py` - набор тестов для интегрирования с постоянным шагом:
		* `test_one_step()` - проверит одношаговые методы
		* `test_multi_step()` - проверит многошаговые методы

	* `explicit_one_step_methods.py` - тут должны быть ваши одношаговые методы:
		* `euler()` - метод Эйлера (уже реализован)
		* `runge_kutta()` - метод Рунге-Кутты 

	* `explicit_multistep_methods.py` - тут должны быть ваши многошаговые методы:
		* `adams()` - метод Адамса 

	* `fix_step_integration.py` - тут алгоритм интегрирования с постоянным шагом:
		* `fix_step_integration()` - он самый (уже реализован)
