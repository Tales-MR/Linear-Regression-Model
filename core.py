import numpy


class LinearRegression():

    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y
         
        # # __ after "self." and before our variable' name makes it private
        self.__correlation_coefficient = self.__calculate_correlation()
        self.__inclination = self.__calculate_inclination()
        self.__intercept = self.__calculate_intercept()



    def __calculate_correlation(self):
        x_to_calculate =  self.x
        y_to_calculate =  self.y

        lenght = len(x_to_calculate)

        cov = 0

        var_x = 0
        var_y = 0

        for i in range(lenght):
            x_calculated = (x_to_calculate[i] - numpy.mean(x_to_calculate))
            y_calculated = (y_to_calculate[i] - numpy.mean(y_to_calculate))


            cov += x_calculated * y_calculated

            var_x += (x_calculated) ** 2
            var_y += (y_calculated) ** 2

        cov = cov / lenght

        var_x = var_x / lenght
        var_y = var_y / lenght

        var = numpy.sqrt(var_x * var_y) 

        out = cov / var
    
        return out



    def __calculate_inclination(self):
        x_to_calculate =  self.x
        y_to_calculate =  self.y

        lenght = len(x_to_calculate)


        sX = 0
        sY = 0

        for i in range(lenght):

            sX += (x_to_calculate[i] - numpy.mean(x_to_calculate)) ** 2
            sY += (y_to_calculate[i] - numpy.mean(y_to_calculate)) ** 2

        sX = numpy.sqrt(sX / (lenght -1))
        sY = numpy.sqrt(sY / (lenght -1))

        out = self.__correlation_coefficient * (sY / sX)

        return out


    def __calculate_intercept(self):
        x_to_calculate = self.x
        y_to_calculate = self.y

        x = numpy.mean(x_to_calculate)
        y = numpy.mean(y_to_calculate)

        out = y - (self.__inclination * x)

        return out

    def predict(self, value):
        
        b = self.__intercept
        m = self.__inclination

        out = float(b + (m * value))

        return out


    


x = [18, 23, 25]
y = [158, 298, 315]

ModelLinearRegression = LinearRegression(x, y)

ModelLinearRegression.predict(24)

304.0