#### Question 1a.

$$
\mathbf{S} = \left[\begin{array}
{rr}
10.9 & -12.7 \\
-12.7 & 26.7
\end{array}\right]
$$

$$
\begin{array}{ll}
r_t = \text{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\
z_t = \text{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\
n_t = \text{tanh}(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{t-1} + b_{hn})) \\
h_t = (1 - z_t) * n_t + z_t + h_{t-1} \\
\end{array}
$$



```python
mylist = [1,2,3,4]
print(mylist)
```

    [1, 2, 3, 4]


This is a formula here:
$$
x^2 + y^3 + 4*zyx
$$


```python
mylist.append(5)
mylist
```




    [1, 2, 3, 4, 5]




```python
print("hi my name is giraffe i like to pirouette during nutracker ballet")
```

    hi my name is giraffe i like to pirouette during nutracker ballet


$$
\sum_{n = 0}^{\infty} \frac {x^n} {n!}
$$
