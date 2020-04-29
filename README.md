# Manual-Neural-Network
In this notebook we will manually build out a neural network that mimics the TensorFlow API. This will greatly help your understanding when working with the real TensorFlow!

Quick Note on Super() and OOP

class SimpleClass():
    
    def __init__(self,str_input):
        print("SIMPLE"+str_input)

 

class ExtendedClass(SimpleClass):
    
    def __init__(self):
        print('EXTENDED')

 

s = ExtendedClass()

 

class ExtendedClass(SimpleClass):
    
    def __init__(self):
        
        super().__init__(" My String")
        print('EXTENDED')

 

s = ExtendedClass()


Graph
############################################
class Graph():
    
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self

