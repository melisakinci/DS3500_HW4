import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
import matplotlib.colors as colors

SIZE = 300  # The dimensions of the field
OFFSPRING = 2 # Max offspring offspring when a rabbit reproduces
GRASS_RATE = 0.028 # Probability that grass grows back at any location in the next season.
WRAP = True # Does the field wrap around on itself when rabbits move?

class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self, type, size):
        self.size = size
        self.x = rnd.randrange(0, size)
        self.y = rnd.randrange(0, size)
        self.eaten = 0
        self.type = type

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly """

        # change options for moving based on rabbit type
        move_options = [-1, 0, 1]
        if self.type == 1:
            move_options = [-2, -1, 0, 1, 2]

        if WRAP:
            self.x = (self.x + rnd.choice(move_options)) % self.size
            self.y = (self.y + rnd.choice(move_options)) % self.size
        else:
            self.x = min(self.size-1, max(0, (self.x + rnd.choice(move_options))))
            self.y = min(self.size-1, max(0, (self.y + rnd.choice(move_options))))

    def get_type(self):
        """ return type of rabbit """
        return copy.deepcopy(self.type)


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self, size, pygmy = 1, cotton_tail = 1):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.size = size
        self.field = np.ones(shape=(size, size), dtype=int)
        self.rabbits = [Rabbit(0, size) for _ in range(pygmy)] # rabbit 0 is pygmy rabbit
        self.rabbits += [Rabbit(1, size) for _ in range(cotton_tail)] # rabbit 1 is pygmy rabbit
        self.nrabbits = [pygmy + cotton_tail]
        self.ngrass = [size*size]

        self.fig = plt.figure(figsize=(5, 5))
        #plt.title("generation = 0")

        cmap = colors.ListedColormap(['white', 'blue', 'red', 'green'])
        self.im = plt.imshow(self.field, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=1)


    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)

    def move(self):
        """ Rabbits move """
        for r in self.rabbits:
            r.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """

        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x,rabbit.y])
            self.field[rabbit.x,rabbit.y] = 0

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:

            # change number of offspring based on rabit type
            offspring = OFFSPRING
            if rabbit.type == 1:
                offspring == 1
            for _ in range(rnd.randint(1,offspring)):
                born.append(rabbit.reproduce())
        self.rabbits += born

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(self.size, self.size) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(self.size,self.size), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1
        return rabbits

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        return len(self.rabbits)

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

    def animate(self, i, speed=1):
        """ Animate one frame of the simulation"""

        # Run some number of generations before rendering next frame
        for n in range(speed):
            self.generation()

        # Update the frame

        tempfield = self.field

        for rabbit in self.rabbits:
            val = 2
            if rabbit.type == 1:
                val = 3
            tempfield[rabbit.x][rabbit.y] = val


        self.im.set_array(tempfield)

        plt.title("generation = " + str((i+1) * speed))
        return self.im,

    def run(self, generations=10000, speed=1):
        """ Run the simulation. Speed denotes how may generations run between successive frames """
        anim = animation.FuncAnimation(self.fig, self.animate, fargs=(speed,), frames=generations//speed, interval=1, repeat=False)
        plt.show()

    def history(self, showTrack=True, showPercentage=True, marker='.'):
        plt.figure(figsize=(6,6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        xs = self.nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x/maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        ys = self.ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y/maxgrass for y in ys]
            plt.ylabel("% Grass")

        if showTrack:
            plt.plot(xs, ys, marker=marker)
        else:
            plt.scatter(xs, ys, marker=marker)

        plt.grid()

        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def history2(self):

        xs = self.nrabbits[:]
        ys = self.ngrass[:]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=ys, s=5, color=".15")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
        plt.grid()
        plt.xlim(0, max(xs)*1.2)

        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")
        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()


        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def scatter_3d(self):

        # distinguish between type 1 and type 0 rabbits

        xs = len([Rabbit for Rabbit in self.rabbits[:] if Rabbit.type == 0])
        ys = len([Rabbit for Rabbit in self.rabbits[:] if Rabbit.type == 1])
        zs = self.ngrass[:]

        sns.set_style('dark')

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter_3d(x=xs, y=ys, z=zs, s=5, color=".15")
        ax.grid()
        ax.xlim(0, max(xs)*1.2)

        ax.set_xlabel("Pygmy Rabbits")
        ax.set_ylabel("Cottontail Rabbits")
        ax.set_zlabel("Grass")
        ax.title("Pygmy vs. Cottontail vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        ax.savefig("history3d.png", bbox_inches='tight')
        ax.show()


def main():

    print("Please enter the following fallows:\n")
    field_size = input("Field size: \n")
    pygmy = input("Initial Pygmy Rabbit Population: \n")
    cotton_tail = input("Initial Cotton Tail Population: \n")
    speed = input("Speed of simulation (generations per frame): \n")

    # Create the ecosystem
    field = Field(pygmy = int(pygmy), cotton_tail = int(cotton_tail), size=  int(field_size))

    # Run the ecosystem
    field.run(generations=5000, speed= int(speed))

    # Plot history
    field.history()
    field.history2()
    field.scatter_3d()


if __name__ == '__main__':
    main()

