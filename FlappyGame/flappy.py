from itertools import cycle
import random
import sys
import itertools
sys.path.append("..")

import pygame
import numpy
from pygame.locals import *

from network import Network


FPS = 200
SCREENWIDTH = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, HITMASKS = {}, {}
BIRDSTOKEEP = 12
BIRDCOUNT = 2*BIRDSTOKEEP + BIRDSTOKEEP*(BIRDSTOKEEP-1)


SIGMA = .03
DISASTER = .005
INNERLAYERS = 5

best_score_ever = 0


# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


class BirdAI:

    def __init__(self, myNet=Network([2, INNERLAYERS, 1]), birdType='mutated'):
        self.neuralNet = myNet
        self.birdType = birdType
        self.imageTup = IMAGES['player'][birdType]
        self.hitmaskTup = HITMASKS['player'][birdType]
        self.playerIndex = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - IMAGES['player'][self.birdType][0].get_height()) / 2) + random.uniform(-20, 20)
        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10   # max vel along Y, max descend speed
        self.playerMinVelY = -8   # min vel along Y, max ascend speed
        self.playerAccY = 1   # players downward accleration
        self.playerRot = 45   # player's rotation
        self.playerVelRot = 3   # angular speed
        self.playerRotThr = 20   # rotation threshold
        self.playerFlapAcc = -9   # players speed on flapping
        self.playerFlapped = False  # when player flaps
        self.fitness = 0
        self.score = 0

    def drawSprite(self):
        SCREEN.blit(self.imageTup[self.playerIndex], (self.playerx, self.playery))

    def advanceIndex(self):
        self.playerIndex = next(self.playerIndexGen)

    def flapWing(self):
        if self.playery > -2 * self.imageTup[0].get_height():
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

    def calculateMovement(self):
        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        self.playerHeight = self.imageTup[self.playerIndex].get_height()
        self.playery += min(self.playerVelY, BASEY - self.playery - self.playerHeight)
        self.fitness += 1

    def rotateImage(self):
        # Player rotation has a threshold
        visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            visibleRot = self.playerRot

        playerSurface = pygame.transform.rotate(self.imageTup[self.playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (self.playerx, self.playery))

    def checkJump(self, xDiff, yDiff):
        out = self.neuralNet.apply(numpy.array([xDiff, yDiff]))

        if(out[0] > 0):
            self.flapWing()

    def birdCrash(self, upperPipes, lowerPipes):
        crash = checkCrash({'x': self.playerx,
                            'y': self.playery,
                            'index': self.playerIndex},
                           self.imageTup,
                           self.hitmaskTup,
                           upperPipes, lowerPipes)
        if crash[0]:
            xMidPos = self.playerx + self.imageTup[0].get_width() / 2
            yMidPos = self.playery + self.imageTup[0].get_height() / 2

            for pipe in upperPipes:
                pipeXMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                pipeYMidPos = pipe['y'] + PIPEGAPSIZE / 2
                if pipeXMidPos > xMidPos:
                    xDiff = pipeXMidPos - xMidPos
                    yDiff = pipeYMidPos - yMidPos
                    self.fitness -= yDiff
                    #self.fitness -= xDiff
                    break
        return crash

    def checkScore(self, upperPipes):
        global best_score_ever
        playerMidPos = self.playerx + self.imageTup[0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.fitness += 300
                self.score += 1
                best_score_ever = max(self.score, best_score_ever)


# main training loop
def main():
    global SCREEN, FPSCLOCK, BIRDPOP
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    BIRDPOP = []
    pygame.display.set_caption('Flappy AI')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[0]).convert()

    # select random pipe sprites
    pipeindex = 0
    # upper pipe, lower pipe
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hitmask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # reset player images
    IMAGES['player'] = {}
    HITMASKS['player'] = {}
    # red => old
    # blue => mated
    # yellow => mutated
    bird_types = ['old', 'mated', 'mutated']

    for i in range(0, 3):

        # load all image types for bird types
        IMAGES['player'][bird_types[i]] = (
            pygame.image.load(PLAYERS_LIST[i][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[i][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[i][2]).convert_alpha(),
        )

        # load all hitmasks for each bird
        HITMASKS['player'][bird_types[i]] = (
            getHitmask(IMAGES['player'][bird_types[i]][0]),
            getHitmask(IMAGES['player'][bird_types[i]][1]),
            getHitmask(IMAGES['player'][bird_types[i]][2]))

    for i in range(0, BIRDCOUNT):
        BIRDPOP.append(BirdAI(birdType='mutated'))

    epoch = 0

    # Paint all 10 birds on the screen, wait for user to press enter
    movementInfo = showWelcomeAnimation(BIRDPOP)

    # epoch loop
    while True:

        print("Epoch: {}\tBest: {}".format(epoch, best_score_ever))
        # main game will run all of the birds until failure, then
        mainGame(movementInfo, BIRDPOP)
        BIRDPOP = sorted(BIRDPOP, key=lambda bird: -(bird.fitness))
        liveBirdsLeft = BIRDPOP[0:BIRDSTOKEEP]

        newPop = []
        # add the newest birds to the population
        for bird in liveBirdsLeft:
            newPop.append(BirdAI(myNet=bird.neuralNet, birdType='old'))

        newPop += [BirdAI(Network.mate(a.neuralNet, b.neuralNet), birdType='mated')
                   for a, b in itertools.combinations(liveBirdsLeft, 2)]

        for i in range(0, len(newPop)):
            newPop.append(BirdAI(myNet=newPop[i].neuralNet.mutate(SIGMA, DISASTER),
                                 birdType='mutated'))

        BIRDPOP = newPop
        print(len(newPop))
        epoch += 1

        # Paint all 10 birds on the screen, wait for user to press enter
        movementInfo = showWelcomeAnimation(BIRDPOP, noSpace=True)



def showWelcomeAnimation(BIRDPOP,noSpace=False):
    """Shows welcome screen animation of flappy bird"""

    basex = 0
    loopIter = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                # make first flap sound and return values for mainGame
                '''
                    The movement info to return
                    playery is initial position + current shm (cycles through -8 to 8)
                    basex is the x pos of player?
                    playerIndexGen is the current wing flap position
                '''
                return {
                    'basex': basex
                }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            for bird in BIRDPOP:
                bird.advanceIndex()

        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        for bird in BIRDPOP:
            bird.drawSprite()

        pygame.display.update()
        FPSCLOCK.tick(FPS)

        if noSpace:
            return {
                'basex': basex
            }


def mainGame(movementInfo, BIRDPOP):
    random.shuffle(BIRDPOP)

    score = loopIter = 0
    birdsLeft = len(BIRDPOP)
    # playerIndexGen = movementInfo['playerIndexGen']
    # playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 50, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 50 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 50, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 50 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    #print(upperPipes)
    #print(lowerPipes)

    pipeVelX = -4

    while True:

        for event in pygame.event.get():
            0  # ignore all events

        for bird in BIRDPOP:
            xMidPos = bird.playerx + bird.imageTup[0].get_width() / 2
            yMidPos = bird.playery + bird.imageTup[0].get_height() / 2

            # find the first pipe past the bird, calculate distances to the pipe
            for pipe in upperPipes:
                pipeXMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                pipeYMidPos = pipe['y'] + PIPEGAPSIZE / 2
                if pipeXMidPos + IMAGES['pipe'][0].get_width() / 2 > xMidPos - bird.imageTup[0].get_width() / 2:
                    xDiff = pipeXMidPos - xMidPos
                    yDiff = pipeYMidPos - yMidPos
                    break

            # have the bird try to jump
            bird.checkJump(xDiff, yDiff)

        liveBirds = []
        # check for crash here
        for bird in BIRDPOP:
            score = max(score, bird.score)
            crashTest = bird.birdCrash(upperPipes, lowerPipes)

            if not crashTest[0]:
                liveBirds.append(bird)
            else:
                birdsLeft -= 1
                #print("BIRDS LEFT: {}".format(birdsLeft))

        BIRDPOP = liveBirds

        # check for score
        for bird in BIRDPOP:
            bird.checkScore(upperPipes)

        # return fittest members of population
        if birdsLeft == 0:
            return

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            for bird in BIRDPOP:
                bird.advanceIndex()

        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        for bird in BIRDPOP:
            bird.calculateMovement()

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        #print(upperPipes)
        #print(lowerPipes)
        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            #print('Creating new pipe')
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        for bird in BIRDPOP:
            bird.rotateImage()

        #print("UPdate")
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)


        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, imageTup, hitmaskTup, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = imageTup[0].get_width()
    player['h'] = imageTup[0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = hitmaskTup[pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
