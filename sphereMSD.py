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
n_random_walks = 15000
sequence_displacements = []
for j in range(0,21,1):
    n_steps = j
    displacements = []
    for i in range(0,n_random_walks):
        a_walk = rwalk(north_pole, n_steps)
        dist = math.acos(dot(a_walk[0],a_walk[-1]))
        displacements.append(dist**2)
    sequence_displacements.append(np.average(displacements))
    print(len(sequence_displacements))

horizontal = np.arange(len(sequence_displacements))
a_fit = -2.93
b_fit = -0.385
c_fit = 2.93
y_fit = exp_func(horizontal, a_fit, b_fit, c_fit)
print(y_fit)
print(sequence_displacements)
plt.plot(horizontal,sequence_displacements, marker='o', label='MSD 15,000 random walks')
plt.plot(horizontal,y_fit,label='Fitted Exponential b=-0.386')
plt.legend()
plt.xlabel('Number of steps')
plt.ylabel('Mean Squared Displacement')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
file_name = 'msdhist' + str(n_random_walks) + '.png'
plt.savefig(file_name)
plt.show()
