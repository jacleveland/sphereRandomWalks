import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

sample_length = 2000

def inv_cdf(x):
    result = []
    for entry in x:
        result.append(math.sqrt((2*entry)/(math.pi-2)))
    return result

def tangent_vector(angle,magnitude):
    result = []
    result.append(magnitude * math.cos(angle))
    result.append(magnitude * math.sin(angle))
    return result

def magnitude(vector):
    result = 0
    for component in vector:
        result += component**2
    return math.sqrt(result)

def scale(vector,scalar):
    result = []
    for component in vector:
        result.append(scalar * component)
    return result

'''
Embed 2d sampled vector in 3d roughly aligned with tangent space at point 
Point is the current point, where the tangent space is we care about
Vector is the sampled 2d tangent vector
Insert a 0 into the component based on where on the sphere we are
Without doing this, the 2d vectors are too perpendicular to high x and high y tangent planes.
'''
def embed(point, vector):
    temp = []
    for coord in point:
        temp.append(abs(coord))
    max_value = max(temp)
    max_index = temp.index(max_value)
    if(max_index == 0):
        vector.insert(0,0)
    elif(max_index == 1):
        vector.insert(1,0)
    elif(max_index == 2):
        vector.insert(2,0)
    return vector

'''
Project 3d vector 'vector' onto tangent space at 'point'
using linear algebra, proj(v)=v-(n*v/|n|)n
'''
def project(point,vector):
    result = []
    original_magnitude = magnitude(vector)
    n_dot_v = point[0]*vector[0]+point[1]*vector[1]+point[2]*vector[2]
    n_scale = n_dot_v/(magnitude(point))
    scaled_n = scale(point,n_scale)
    projected_vector = []
    for i in range(0,len(vector)):
        projected_vector.append(vector[i]-scaled_n[i])
    magnitude_ratio = original_magnitude/magnitude(projected_vector)
    result = scale(projected_vector,magnitude_ratio)
    return result

'''
Component-wise addition of two vectors of arbitrary but equal length.
'''
def add(vec1,vec2):
    result = []
    for i in range(0,len(vec1)):
        result.append(vec1[i]+vec2[i])
    return result

'''
Dot product of vectors of arbitrary but equal length.
'''
def dot(vec1,vec2):
    result = 0
    for i in range(0,len(vec1)):
        result += vec1[i]*vec2[i]
    return result

'''
Exponential map exp:Tp(S^2)->S^2

Maps a tangent vector at point p to another point on the sphere.
Well defined since the sphere is a complete manifold.
point is a point on the sphere, vector is tangent vector.
Returns gamma(1)=cos(|v|)p+sin(|v|)v/|v|
'''
def exp_map(point,vector):
    result = []
    tangent_magnitude = magnitude(vector)
    new_point = scale(point,math.cos(tangent_magnitude))
    vertical_adjust = scale(vector,(math.sin(tangent_magnitude)/tangent_magnitude))
    result = add(new_point,vertical_adjust)
    return result

'''
exponential function
'''
def exp_func(x, a, b, c):
    return c+a*np.exp(b*x)

'''
Return a sequence of steps of length n_steps starting at point start on sphere.
'''
def rwalk(start, n_steps):
    result = [start]
    angle_sample = np.random.uniform(0,2*math.pi,n_steps).tolist()
    magnitude_sample = np.sqrt(np.random.uniform(0,math.pi/2,n_steps).tolist())
    sample = np.array([angle_sample,magnitude_sample])
    sample = sample.transpose()
    sample = sample.tolist()
    tangent_vectors = []
    for entry in sample:
        tangent_vectors.append(tangent_vector(entry[0],entry[1]))
    current_point = start
    exponentiated = []
    for vec in tangent_vectors:
        vec = embed(current_point,vec)
        projected_tangent = project(current_point,vec)
        next_point = exp_map(current_point,projected_tangent)
        current_point = next_point
        result.append(current_point)
    return result

north_pole = [0,0,1]
colors = ['r','orange','b','g']
a_walk = []
n_steps = 20
for i in range(0,1):
    a_walk.append(rwalk(north_pole, n_steps))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
count = 0
for walk in a_walk:
    x = []
    y = []
    z = []
    for entry in walk:
        x.append(entry[0])
        y.append(entry[1])
        z.append(entry[2])
    ax.plot(x,y,z,c=colors[count],marker='o')
    count += 1
plt.savefig('1walk.png')
plt.show()
