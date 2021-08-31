# ------------------------- IMPORTS -------------------------

import cv2
import time
import numpy as np
import math
import os
import sys
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import colors

# -------------------- GLOBAL VARIABLES ---------------------
LEFT = 0
STRAIGHT = 1
RIGHT = 2
SPEEDR = 200
SPEEDL = 200
TIME_UNIT = 0.268
TIME_UNIT_RIGHT_TURN = 0.635
TIME_UNIT_LEFT_TURN = 0.635

# ------------------------ FUNCTIONS ------------------------

# 2.1 A* Algorithm
def createEmptyPlot(gridSize):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param grid_size: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks = np.arange(0, gridSize+1, 5)
    minor_ticks = np.arange(0, gridSize+1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,gridSize])
    ax.set_xlim([-1,gridSize])
    ax.grid(True)
    
    return fig, ax

def getMovements():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.9),
            (-1, 1, 1.9),
            (-1, -1, 1.9),
            (1, -1, 1.9)]

def reconstructPath(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    totalPath = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        totalPath.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    return totalPath

def AStar(start, goal, h, coords, occupancyGrid, gridSize):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    for point in [start, goal]:
        for coord in point:
            assert coord>=0 and coord<gridSize, "start or end goal not contained in the map"
            
    if occupancyGrid[start[0], start[1]]:
        raise Exception('Start node is not traversable')
    if occupancyGrid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
        
    movements = getMovements()
    openSet = [start]
    closedSet = []
    cameFrom = dict()
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    while openSet != []:      
        fScoreOpenSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScoreOpenSet, key=fScoreOpenSet.get)
        del fScoreOpenSet
        
        if current == goal:
            return reconstructPath(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        for dx, dy, deltacost in movements:
            neighbor = (current[0]+dx, current[1]+dy)
            if (neighbor[0] >= occupancyGrid.shape[0]) or (neighbor[1] >= occupancyGrid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            if (occupancyGrid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue            
            tentativeGScore = gScore[current] + deltacost
            if neighbor not in openSet:
                openSet.append(neighbor)
            if tentativeGScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentativeGScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]
                
    print("No path found to goal")
    return [], closedSet

def add_thymio_obstacle(pathCoords_green, obstacleGrid_green):
    #the middle of green thymio path become an obstacle for the red one
    obstacleGrid_red=obstacleGrid_green.copy()
   
    x,y=pathCoords_green[math.floor(len(pathCoords_green)/2)]
    #print(x,y)
    cv2.rectangle(obstacleGrid_red, (y-5,x-5), (y+5,x+5) , (1), -1)
  
     # Applying morphological operators to the image
   
    occupancyGrid_red = cv2.dilate(obstacleGrid_red, None, iterations=3)
    
    return occupancyGrid_red

def runAStarSingle(gridSize, obstacleGrid, occupancyGrid, start, goal, show):
    # Norm parameters
    norm_ord = 1
    norm_mul = 2
    
    # List of all coordinates in the grid
    x,y = np.mgrid[0:gridSize:1, 0:gridSize:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1, ord=norm_ord ) * norm_mul
    h = dict(zip(coords, h))
    
    # Run the A* algorithm
    pathCoords, visitedNodes = AStar(start, goal, h, coords, occupancyGrid, gridSize)
    path = np.array(pathCoords).reshape(-1, 2).transpose()
    
    # Displaying the map
    if show == 1:
        figAStar, axAStar = createEmptyPlot(gridSize)
    else:
        axAStar = 0
    cmap = colors.ListedColormap(['white', 'slategray'])
    
    return pathCoords, path, axAStar, cmap

def runAStarSwitch(gridSize, obstacleGrid, occupancyGrid, start, goal, show):
    # Norm parameters
    norm_ord1 = 1
    norm_mul1 = 2
    
    norm_ord2 = 1
    norm_mul2 = 2

    # List of all coordinates in the grid
    x,y = np.mgrid[0:gridSize:1, 0:gridSize:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles
    h1 = np.linalg.norm(pos - goal, axis=-1, ord=norm_ord1 ) * norm_mul1
    h1 = dict(zip(coords, h1))
    
    h2 = np.linalg.norm(pos - start, axis=-1, ord=norm_ord2 ) * norm_mul2
    h2 = dict(zip(coords, h2))

    # Run the A* algorithm
    pathCoordsG, visitedNodesG = AStar(start, goal, h1, coords, occupancyGrid, gridSize)    
    
    occupancyGridR = add_thymio_obstacle(pathCoordsG, obstacleGrid)
    pathCoordsR, visitedNodesR = AStar(goal, start, h2, coords, occupancyGridR, gridSize)    

    pathG = np.array(pathCoordsG).reshape(-1, 2).transpose()
    visitedNodesG = np.array(visitedNodesG).reshape(-1, 2).transpose()   
    pathR = np.array(pathCoordsR).reshape(-1, 2).transpose()
    visitedNodesR = np.array(visitedNodesR).reshape(-1, 2).transpose()

    # Displaying the map
    if show == 1:
        figAStar, axAStar = createEmptyPlot(gridSize)
        cmap = colors.ListedColormap(['white', 'slategray'])
    else:
        axAStar = 0
    
    return pathCoordsG, pathG, pathCoordsR, pathR, axAStar, cmap


# 2.2 PARAMETER FUNCTIONS
def getMotionDicts():
    t1,t2 = np.mgrid[-1:2:1, -1:2:1]
    tp = np.empty(t1.shape + (2,))
    tp[:, :, 0] = t1; tp[:, :, 1] = t2
    tp = np.reshape(tp, (t1.shape[0]*t1.shape[1], 2))
    tc = list([(int(t1[0]), int(t1[1])) for t1 in tp])
    tv = [5,6,7,4,8,0,3,2,1]
    deriv2val = dict(zip(tc, tv))
    val2deriv =  dict(zip(tv, tc))
    return deriv2val, val2deriv

DERIV2VAL, VAL2DERIV = getMotionDicts()

def getParameters(th, orientation, path):
    """
    Takes a list of path coordinates and converts it into an array of movements
    :path:   List[(x,y), ... (x,y)]
    :return: Array[ [x y orientation] ... [x y orientation] ], Array[ [dx dy] ... [dx dy] ] 
    """
    # Create an array of the planned movements [dx dy]
    movArray = np.empty((len(path)-1,2))
    for i in range(len(path)):
        if (i != len(path)-1):
            movArray[i][0] = path[i+1][0] - path[i][0]
            movArray[i][1] = path[i+1][1] - path[i][1]
    
    # Get the robots original orientation and face the right way
    orient = np.empty(len(path))
    orient[0] = DERIV2VAL[tuple(movArray[0])]
    updateOrientation(th, orient[0]*45, orientation)
    
    # Create empty direction array
    dirArray = np.empty(len(movArray))
    dirArray[0] = 1
    
    for i in range(len(path)-1):
        orient[i+1] = DERIV2VAL[tuple(movArray[i])]
        if i != 0:
            if (orient[i] - orient[i-1])%8 == 1:
                dirArray[i] = RIGHT
            elif (orient[i] - orient[i-1])%8 == 7:
                dirArray[i] = LEFT
            else:
                dirArray[i] = STRAIGHT
    
    # Create new position coordinates [x y orientation]
    posArray = np.empty((len(path),3))
    
    # Get the robots orientation for the rest of the path coordinates
    for i in range(len(path)):
        posArray[i][0] = path[i][0]
        posArray[i][1] = path[i][1]
        posArray[i][2] = orient[i]
                
    # Return the array of positions and of movements                      
    return posArray, movArray, dirArray
    
def updateCoords(currPos, prevPos, movement, direction):
    """
    Takes in a movement, the robots current position coordinates and an array of its previous coordinates.
    It then updates the current coordinates according to the movement, and updates the previous ones:
    :currCoords:  [x y]
    :prevCoords: [ [x y] ... [x y] ]
    :movement: [dx dy]
    """
    newPrevPos = np.empty((len(prevPos)+1,3))
    newCurrPos = np.empty(3)
    for i in range(len(prevPos)+1):
        if i == (len(prevPos)):
            newPrevPos[i] = currPos
        else:
            newPrevPos[i] = prevPos[i]
    newCurrPos[0] = currPos[0]+movement[0]
    newCurrPos[1] = currPos[1]+movement[1]
    if direction == STRAIGHT:
        newCurrPos[2] = currPos[2]
    elif direction == RIGHT:
        newCurrPos[2] = (currPos[2]+1)%8
    elif direction == LEFT:
        newCurrPos[2] = (currPos[2]-1)%8
        
    return newCurrPos, newPrevPos


# 2.3 NAVIGATION FUNCTIONS
def stop(th):
    """
    Stops the robot motors
    """
    th.set_var_array("leds.circle", [0, 0, 0, 0, 0, 0, 0, 0])
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)

def goForwards(th, unit):
    """
    Starts the robot motors to go straight
    """
    th.set_var_array("leds.circle", [255, 0, 0, 0, 0, 0, 0, 0])
    th.set_var("motor.right.target", SPEEDR)
    th.set_var("motor.left.target", SPEEDL)
    time.sleep(unit*TIME_UNIT)
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)
    
    
def turnRight(th):
    """
    Rotates the robot 45 degrees to the right and sets the motors to go straight
    """
    th.set_var_array("leds.circle", [0, 255, 0, 0, 0, 0, 0, 0])
    th.set_var("motor.right.target", 2**16-SPEEDR)
    th.set_var("motor.left.target", SPEEDL)
    time.sleep(TIME_UNIT_RIGHT_TURN)
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)
    
def turnRightAngle(th, n):
    """
    Rotates the robot 45 degrees to the right and sets the motors to go straight
    """
    th.set_var_array("leds.circle", [0, 255, 0, 0, 0, 0, 0, 0])
    th.set_var("motor.right.target", 2**16-SPEEDR)
    th.set_var("motor.left.target", SPEEDL)
    time.sleep(TIME_UNIT_RIGHT_TURN*n/45)
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)
        
def turnLeft(th):
    """
    Rotates the robot 45 degrees to the left and sets the motors to go straight
    """
    th.set_var_array("leds.circle", [0, 0, 0, 0, 0, 0, 0, 255])
    th.set_var("motor.right.target", SPEEDR)
    th.set_var("motor.left.target", 2**16-SPEEDL)
    time.sleep(TIME_UNIT_LEFT_TURN)
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)
    
def turnLeftAngle(th, n):
    """
    Rotates the robot 45 degrees to the right and sets the motors to go straight
    """
    th.set_var_array("leds.circle", [0, 255, 0, 0, 0, 0, 0, 0])
    th.set_var("motor.right.target", SPEEDR)
    th.set_var("motor.left.target", 2**16-SPEEDL)
    time.sleep(TIME_UNIT_LEFT_TURN*n/45)
    th.set_var("motor.right.target", 0)
    th.set_var("motor.left.target", 0)
    
def updateOrientation(th, orient, currOrient):
    n = abs(currOrient-orient)
    if (orient < currOrient):
        if (n) > 180:
            n = 360-n%360
            right = True
        else:
            right = False
    else:
        if (n) > 180:
            n = 360-n%360
            right = False
        else:
            right = True        
    if right:
        turnRightAngle(th, n)
        print(n, "degree right turn")
    else:
        turnLeftAngle(th, n)  
        print(n, "degree left turn")

def getMovementNorm(movement):
    """
    Sleeps for a time that allows forward movement
    """
    return math.sqrt(movement[0]**2+movement[1]**2)
    
def globalNavigation(th, movement, direction):
    """
    Takes in a movement and direction and applies the specific motor instruction:
    :movement:  [dx dy]
    :direction: [d]
    """
    unit = getMovementNorm(movement)
    if direction == STRAIGHT:
        goForwards(th, unit)
    elif direction == RIGHT:
        turnRight(th)
        goForwards(th, unit)
    elif direction == LEFT:
        turnLeft(th)
        goForwards(th, unit)
        
        
def angleX(x, y):
    if (x==0 and y==0):
        angle = 0
    elif (x == 0):
        if y < 0:
            angle = 90
        else:
            angle = 270
    else:
        if x > 0:
            angle = (np.arctan(y/x)*180/np.pi)%360
        else:
            angle = (180+np.arctan(y/x)*180/np.pi)%360
            
    return angle


def correctPosition(th, wantedPos, realPos):
    # Get trajectory vector
    v = [wantedPos[0]-realPos[0], wantedPos[1]-realPos[1]]
    
    # Get vector angle and euclidean norm
    vAngle = angleX(v[0], v[1])
    vNorm = math.sqrt(v[0]**2 + v[1]**2)
    
    if (vNorm == 0 and vAngle == 0):
        updateOrientation(th, wantedPos[2]*45, realPos[2])
    else:
        # Orient robot with vector
        updateOrientation(th, vAngle, realPos[2])

        # Go to wanted point  
        goForwards(th, TIME_UNIT*vNorm)
        stop(th)

        # Update orientationt to resume path
        updateOrientation(th, wantedPos[2]*45, vAngle)  
    