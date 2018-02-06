#!/usr/bin/env python

#
#   Breakout V 0.1 June 2009
#
#   Copyright (C) 2009 John Cheetham    
#
#   web   : http://www.johncheetham.com/projects/breakout
#   email : developer@johncheetham.com
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#    
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import pygame
import random

import relearn.breakout.brain.cnn.nn as cnn
import relearn.breakout.brain.nn.nn_brain as nn


class Breakout:
    def __init__(self, brain):
        self.brain = brain

    def main(self):
        self.brain.start()
        xspeed_init = 6
        yspeed_init = 6
        max_lives = 5
        bat_speed = 30
        score = 0
        bgcolour = 0x2F, 0x4F, 0x4F  # darkslategrey        
        size = width, height = 600, 600

        pygame.init()
        screen = pygame.display.set_mode(size)
        # screen = pygame.display.set_mode(size, pygame.FULLSCREEN)

        bat = pygame.image.load("bat.png").convert()
        batrect = bat.get_rect()

        ball = pygame.image.load("ball.png").convert()
        ball.set_colorkey((255, 255, 255))
        ballrect = ball.get_rect()

        pong = pygame.mixer.Sound('Blip_1-Surround-147.wav')
        pong.set_volume(0)

        wall = Wall()
        wall.build_wall(width)

        # Initialise ready for game loop
        batrect = batrect.move((width / 2) - (batrect.right / 2), height - 20)
        ballrect = ballrect.move(width / 2, height / 2)
        xspeed = xspeed_init
        yspeed = yspeed_init
        lives = max_lives
        clock = pygame.time.Clock()
        pygame.key.set_repeat(1, 30)
        pygame.mouse.set_visible(0)  # turn off mouse pointer

        while 1:

            # 60 frames per second
            clock.tick(100)

            move = self.choose_action()

            batrect = batrect.move(move * bat_speed, 0)

            if move == -1 and batrect.left < 0:
                batrect.left = 0
            elif move == 1 and batrect.right > width:
                batrect.right = width

            # check if bat has hit ball    
            if batrect.top <= ballrect.bottom <= batrect.bottom and \
                            ballrect.right >= batrect.left and \
                            ballrect.left <= batrect.right:
                yspeed = -yspeed
                pong.play(0)
                self.brain.new_score(0.7)
                offset = ballrect.center[0] - batrect.center[0]
                # offset > 0 means ball has hit RHS of bat                   
                # vary angle of ball depending on where ball hits bat                      
                if offset > 0:
                    if offset > 30:
                        xspeed = 7
                    elif offset > 23:
                        xspeed = 6
                    elif offset > 17:
                        xspeed = 5
                else:
                    if offset < -30:
                        xspeed = -7
                    elif offset < -23:
                        xspeed = -6
                    elif xspeed < -17:
                        xspeed = -5

                        # move bat/ball
            ballrect = ballrect.move(xspeed, yspeed)
            if ballrect.left < 0 or ballrect.right > width:
                xspeed = -xspeed
                pong.play(0)
            if ballrect.top < 0:
                yspeed = -yspeed
                pong.play(0)

                # check if ball has gone past bat - lose a life
            if ballrect.top > height:
                lives -= 1
                self.brain.new_score(0)
                # start a new ball
                xspeed = xspeed_init
                rand = random.random()
                if random.random() > 0.5:
                    xspeed = -xspeed
                yspeed = yspeed_init
                ballrect.center = width * random.random(), height / 3
                if lives == 0:
                    self.brain.end()
                    return

            if xspeed < 0 and ballrect.left < 0:
                xspeed = -xspeed
                pong.play(0)

            if xspeed > 0 and ballrect.right > width:
                xspeed = -xspeed
                pong.play(0)

            # check if ball has hit wall
            # if yes yhen delete brick and change ball direction
            index = ballrect.collidelist(wall.brickrect)
            if index != -1:
                if ballrect.center[0] > wall.brickrect[index].right or \
                                ballrect.center[0] < wall.brickrect[index].left:
                    xspeed = -xspeed
                else:
                    yspeed = -yspeed
                pong.play(0)
                wall.brickrect[index:index + 1] = []
                score += 10
                self.brain.new_score(1.0)

            screen.fill(bgcolour)
            scoretext = pygame.font.Font(None, 40).render(str(score), True, (0, 255, 255), bgcolour)
            scoretextrect = scoretext.get_rect()
            scoretextrect = scoretextrect.move(width - scoretextrect.right, 0)
            screen.blit(scoretext, scoretextrect)

            for i in range(0, len(wall.brickrect)):
                screen.blit(wall.brick, wall.brickrect[i])

                # if wall completely gone then rebuild it
            if wall.brickrect == []:
                wall.build_wall(width)
                xspeed = xspeed_init
                yspeed = yspeed_init
                ballrect.center = width / 2, height / 3

            screen.blit(ball, ballrect)
            screen.blit(bat, batrect)
            pygame.display.flip()

    def choose_action(self):
        s = pygame.Surface.copy(pygame.display.get_surface())
        state = pygame.surfarray.pixels3d(s)[:,:,1]
        return self.brain.choose_action(state)


class Wall():
    def __init__(self):
        self.brick = pygame.image.load("brick.png").convert()
        brickrect = self.brick.get_rect()
        self.bricklength = brickrect.right - brickrect.left
        self.brickheight = brickrect.bottom - brickrect.top

    def build_wall(self, width):
        xpos = 0
        ypos = 60
        adj = 0
        self.brickrect = []
        for i in range(0, 52):
            if xpos > width:
                if adj == 0:
                    adj = self.bricklength / 2
                else:
                    adj = 0
                xpos = -adj
                ypos += self.brickheight

            self.brickrect.append(self.brick.get_rect())
            self.brickrect[i] = self.brickrect[i].move(xpos, ypos)
            xpos = xpos + self.bricklength


if __name__ == '__main__':
    # brain = rand.RandomBrain()
    # brain = cnn.ReinforcementLearningBrain()
    brain = nn.SimpleNeuralNetworkBrain()
    br = Breakout(brain)
    i = 0
    while 1:
        i += 1
        print('game %s' % i)
        br.main()
