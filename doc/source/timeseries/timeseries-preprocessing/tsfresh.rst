.. _header-n0:

tsfresh
=======

.. _header-n3:

tsfresh 安装
------------

.. code:: shell

   pip install tsfresh

.. _header-n5:

Example
-------

data

.. code:: python

   from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
   from tsfresh.examples.robot_execution_failures import load_robot_execution_failures

   download_robot_execution_failures()
   timeseries, y = load_robot_execution_failures()

.. _header-n9:

安装
----

.. code:: shell

   $ pip install tsfresh

.. code:: python

   from tsfresh.examples.robot_execution_failures import down_robot_execution_failures, load_robot_execution_failures
   down_robot_execution_failures()
   timeseries, y = load_robot_execution_failures()
   print(timeseries.head(20))
   print(y.head(20))

.. _header-n14:

tsfresh 数据格式
----------------

**输入数据格式：**

-  Flat DataFrame

-  Stacked DataFrame

-  dictionary of flat DataFrame

+-----------+--------------+-------------+-------------+
| column_id | column_value | column_sort | column_kind |
+===========+==============+=============+=============+
| id        | value        | sort        | kind        |
+-----------+--------------+-------------+-------------+

.. _header-n34:

Flat DataFrame
~~~~~~~~~~~~~~

\|id \|time \|x \|y\| \| | |  |-\| \|A \|t1 \|x(A, t1) \|y(A, t1)
\|A \|t2 \|x(A, t2) \|y(A, t2) \|A \|t3 \|x(A, t3) \|y(A, t3) \|B \|t1
\|x(B, t1) \|y(B, t1) \|B \|t2 \|x(B, t2) \|y(B, t2) \|B \|t3 \|x(B, t3)
\|y(B, t3)

.. _header-n36:

Stacked DataFrame
~~~~~~~~~~~~~~~~~

\|id \|time|kind|value\| \| | | | \| \|A \|t1 \|x \|x(A, t1)
\|A \|t2 \|x \|x(A, t2) \|A \|t3 \|x \|x(A, t3) \|A \|t1 \|y \|y(A, t1)
\|A \|t2 \|y \|y(A, t2) \|A \|t3 \|y \|y(A, t3) \|B \|t1 \|x \|x(B, t1)
\|B \|t2 \|x \|x(B, t2) \|B \|t3 \|x \|x(B, t3) \|B \|t1 \|y \|y(B, t1)
\|B \|t2 \|y \|y(B, t2) \|B \|t3 \|y \|y(B, t3)

.. _header-n38:

Dictionary of flat DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{ “x”:

+----+------+----------+
| id | time | value    |
+====+======+==========+
| A  | t1   | x(A, t1) |
+----+------+----------+
| A  | t2   | x(A, t2) |
+----+------+----------+
| A  | t3   | x(A, t3) |
+----+------+----------+
| B  | t1   | x(B, t1) |
+----+------+----------+
| B  | t2   | x(B, t2) |
+----+------+----------+
| B  | t3   | x(B, t3) |
+----+------+----------+

, “y”:

+----+------+----------+
| id | time | value    |
+====+======+==========+
| A  | t1   | y(A, t1) |
+----+------+----------+
| A  | t2   | y(A, t2) |
+----+------+----------+
| A  | t3   | y(A, t3) |
+----+------+----------+
| B  | t1   | y(B, t1) |
+----+------+----------+
| B  | t2   | y(B, t2) |
+----+------+----------+
| B  | t3   | y(B, t3) |
+----+------+----------+

}

**输出数据格式：**

+----+-----------------+---+-----------------+-----------------+---+-----------------+
| id | x\ *feature*\ 1 | … | x\ *feature*\ N | y\ *feature*\ 1 | … | y\ *feature*\ N |
+====+=================+===+=================+=================+===+=================+
| A  | …               | … | …               | …               | … | …               |
+----+-----------------+---+-----------------+-----------------+---+-----------------+
| B  | …               | … | …               | …               | … | …               |
+----+-----------------+---+-----------------+-----------------+---+-----------------+
