class Circle():
    def __init__(self):
        self.radius=None

#Return true if the object argument is an instance of the classinfo arguent
#or of a (direct,indirect or virtual) subclass thereof
        
    circle = Circle()
    if isinstance(circle, Circle):
        print('Yes')
    
#Returns True if an object contains specific argument, and Fse otherwise   
    
    circle = Circle()
    if hasattr(circle,'radius'):
        print('Yes')
    
#Runs a line as Python code
    
    statement='''a=10; print(a+5)'''
    exec(statement)