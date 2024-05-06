import copy
import math
import random
import sys
import time
from functools import lru_cache
import pygame
import torch

from config import *

class GameObject:
    def __init__(self, x, y, image):
        self.rect = pygame.Rect(x, y, image.get_width(), image.get_height())
        self.image = image
        self.health = 200
        self.max_health = self.health

    def draw_health_bar(self, screen, x, y, width, height, color=(255, 0, 0)):
        pygame.draw.rect(screen, color, (x, y, width * (self.health / self.max_health), height))

    def draw_debug(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect, 2)  # 2 is thickness of the line


class Player(GameObject):
    def __init__(self, x, y, image, primary_bullet_image, secondary_bullet_image,
                 full_hp_bar=None, full_dash_bar=None, full_secondary_bar=None, empty_bar=None,
                 speed=PLAYER_SPEED, health=MAX_PLAYER_HEALTH, bullet_path='normal_function', primary_shot_sound=None, rendering=True):
        super().__init__(x, y, image)
        self.rendering = rendering
        self.primary_bullet_image = primary_bullet_image
        self.secondary_bullet_image = secondary_bullet_image
        self.speed = speed * 16.666
        self.health = health
        self.max_health = health
        self.camera = Camera(MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.bullets = []
        self.dashing = False
        self.secondary_cooldown = SECONDARY_PROJECTILE_COOLDOWN
        self.dash_duration_remaining = DASH_DURATION
        self.float_x = x
        self.float_y = y
        self.double_shot_active = False
        self.coins = 0
        self.percent_hp_gain = 0
        self.primary_damage = PLAYER_DAMAGE
        self.primary_bullet_piercing = False
        self.primary_shot_sound = primary_shot_sound
        if primary_shot_sound:
            primary_shot_sound.set_volume(0.0)
        self.homing_factor = 0
        self.primary_bullet_size = 1
        self.bullet_path = bullet_path
        self.projectile_timer = PROJECTILE_COOLDOWN
        self.primary_projectile_cooldown = PROJECTILE_COOLDOWN
        self.secondary_projectile_timer = SECONDARY_PROJECTILE_COOLDOWN
        self.dash_timer = DASH_COOLDOWN
        if self.rendering:
            font = pygame.font.SysFont('Arial', 16, bold=True, italic=False)
            self.dash_text = font.render('Dash', True, (255, 255, 255))
            self.secondary_text = font.render('Secondary Fire', True, (255, 255, 255))
            self.health_text = font.render('Health', True, (255, 255, 255))
        if full_hp_bar:
            self.full_hp_bar = pygame.transform.scale(full_hp_bar,
                                                      (full_hp_bar.get_width() * 4, full_hp_bar.get_height() * 4))
        if full_dash_bar:
            self.full_dash_bar = pygame.transform.scale(full_dash_bar,
                                                        (full_dash_bar.get_width() * 4, full_dash_bar.get_height() * 4))
        if full_secondary_bar:
            self.full_secondary_bar = pygame.transform.scale(full_secondary_bar, (
            full_secondary_bar.get_width() * 4, full_secondary_bar.get_height() * 4))
        if empty_bar:
            self.empty_bar = pygame.transform.scale(empty_bar, (empty_bar.get_width() * 4, empty_bar.get_height() * 4))

    def move(self, action, delta_time):

        # print(action)
        keys, mouse_buttons, mouse_x, mouse_y = action
        # each key is true or false (1,0)
        w, a, s, d, space = keys["w"], keys["a"], keys["s"], keys["d"], keys["space"]

        if self.dashing:
            # Dash movement
            time_to_dash = min(self.dash_duration_remaining, delta_time)
            self.dash_duration_remaining -= time_to_dash
            dash_dx = (int(d) - int(a)) * self.speed * DASH_MULTIPLIER * time_to_dash
            dash_dy = (int(s) - int(w)) * self.speed * DASH_MULTIPLIER * time_to_dash
        else:
            dash_dx = 0
            dash_dy = 0
            time_to_dash = 0
            self.dash_timer -= delta_time

        # Normal movement
        time_to_move = delta_time - time_to_dash
        move_dx = (int(d) - int(a)) * self.speed * time_to_move
        move_dy = (int(s) - int(w)) * self.speed * time_to_move

        # Sum the dash and normal movements
        dx = dash_dx + move_dx
        dy = dash_dy + move_dy
        bullets = []
        if self.can_dash() and space:
            bullets = self.start_dash()

        self.float_x += dx
        self.float_y += dy

        new_x = max(0, min(MAP_WIDTH - self.rect.width, self.float_x))
        new_y = max(0, min(MAP_HEIGHT - self.rect.height, self.float_y))

        self.rect.x = int(new_x)
        self.rect.y = int(new_y)
        self.float_x = new_x
        self.float_y = new_y

        if self.dashing and self.dash_duration_remaining <= 0:
            self.dashing = False
        return bullets

    def fire(self, action, delta_time):
        # Extract the actions from the tuple
        keys, mouse_buttons, mouse_x, mouse_y = action

        bullets = []
        if mouse_buttons[0] and self.projectile_timer <= 0:
            # print(self.primary_shot_sound)
            if self.primary_shot_sound:
                self.primary_shot_sound.play()
            self.projectile_timer = self.primary_projectile_cooldown
            player_screen_pos = self.camera.apply(self)

            mouse_pos = (mouse_x, mouse_y)

            direction = calculate_normalized_direction(player_screen_pos, mouse_pos)

            if self.double_shot_active:
                # Calculate the angle of rotation in radians
                angle = math.radians(10)

                # Calculate the rotated directions
                direction_left = (
                    math.cos(angle) * direction[0] - math.sin(angle) * direction[1],
                    math.sin(angle) * direction[0] + math.cos(angle) * direction[1]
                )
                direction_right = (
                    math.cos(-angle) * direction[0] - math.sin(-angle) * direction[1],
                    math.sin(-angle) * direction[0] + math.cos(-angle) * direction[1]
                )

                # Append the bullets with the rotated directions
                bullets.append(
                    Bullet(self.rect.centerx, self.rect.centery, *direction_left, self.primary_bullet_image, speed=1000,
                           damage=self.primary_damage, scale=2 * self.primary_bullet_size, lifetime=.5,
                           bullet_path=self.bullet_path, piercing=self.primary_bullet_piercing,
                           homing_factor=self.homing_factor))
                bullets.append(Bullet(self.rect.centerx, self.rect.centery, *direction_right, self.primary_bullet_image,
                                      speed=1000, damage=self.primary_damage,
                                      lifetime=.5, scale=2 * self.primary_bullet_size, bullet_path=self.bullet_path,
                                      piercing=self.primary_bullet_piercing, homing_factor=self.homing_factor))
            else:
                bullets.append(
                    Bullet(self.rect.centerx, self.rect.centery, *direction, self.primary_bullet_image, speed=1000,
                           damage=self.primary_damage, lifetime=.5, scale=2 * self.primary_bullet_size,
                           bullet_path=self.bullet_path, piercing=self.primary_bullet_piercing,
                           homing_factor=self.homing_factor))
        else:
            self.projectile_timer -= delta_time
        return bullets

    def fire_secondary(self, action, delta_time):
        # Extract the actions from the tuple
        keys, mouse_buttons, mouse_x, mouse_y = action

        bullets = []
        num_shots = 8
        if mouse_buttons[2] and self.secondary_projectile_timer <= 0:
            for i in range(num_shots):
                mouse_pos = self.camera.reverse_apply((mouse_x, mouse_y), self)
                angle = 2 * math.pi / num_shots * i
                direction = math.cos(angle), math.sin(angle)
                bullets.append(
                    Bullet(*mouse_pos, *direction, self.secondary_bullet_image, speed=300, scale=2, damage=15,
                           piercing=True, lifetime=1.5, homing_factor=self.homing_factor))
            self.secondary_projectile_timer = self.secondary_cooldown
        else:
            self.secondary_projectile_timer -= delta_time
        return bullets

    def can_dash(self):
        return self.dash_timer <= 0 and not self.dashing

    def start_dash(self):
        self.dash_timer = DASH_COOLDOWN
        self.dash_duration_remaining = DASH_DURATION  # initialize remaining dash duration
        self.dashing = True
        bullets = self.generate_dash_particles()
        return bullets

    def generate_dash_particles(self):
        # Create new particles
        bullets = []
        for _ in range(20):  # Number of particles generated
            angle = random.uniform(0, 2 * math.pi)
            direction = math.cos(angle), math.sin(angle)
            speed = random.uniform(80, 150)  # pixels per second
            bullets.append(Bullet(self.rect.centerx, self.rect.centery, *direction,
                                  self.primary_bullet_image, speed=speed, scale=1,
                                  damage=0, piercing=False, lifetime=.4))
        return bullets

    def draw_bars(self, screen):
        current_time = pygame.time.get_ticks()
        dash_bar_max_width = 100
        bar_label_y_offset = 25
        bar_x_position = SCREEN_WIDTH - dash_bar_max_width - 70
        bar_y_position = SCREEN_HEIGHT - 85
        bar_spacing = 50

        dash_label_position = (bar_x_position, bar_y_position - bar_label_y_offset)
        secondary_label_position = (bar_x_position, bar_y_position + bar_spacing - bar_label_y_offset)
        health_label_position = (20, SCREEN_HEIGHT - self.full_hp_bar.get_height() - bar_label_y_offset
                                 - self.health_text.get_height())

        screen.blit(self.dash_text, dash_label_position)
        screen.blit(self.secondary_text, secondary_label_position)
        screen.blit(self.health_text, health_label_position)

        # Draw full bars first
        screen.blit(self.full_dash_bar, (bar_x_position, bar_y_position))
        screen.blit(self.full_secondary_bar, (bar_x_position, bar_y_position + bar_spacing))

        # Draw the dash bar
        dash_cooldown_remaining = self.dash_timer
        if dash_cooldown_remaining > 0:
            dash_slice_width = self.empty_bar.get_width() * (dash_cooldown_remaining / DASH_COOLDOWN)
            empty_dash_bar_slice = pygame.Surface((dash_slice_width, self.empty_bar.get_height()), pygame.SRCALPHA)
            empty_dash_bar_slice.blit(self.empty_bar, (0, 0), (
                self.empty_bar.get_width() - dash_slice_width, 0, dash_slice_width, self.empty_bar.get_height()))

            # Set the color key for the slice so black is transparent
            empty_dash_bar_slice.set_colorkey((0, 0, 0))

            screen.blit(empty_dash_bar_slice,
                        (bar_x_position + self.full_dash_bar.get_width() - dash_slice_width, bar_y_position))

        # Draw the secondary bar
        # print(self.secondary_projectile_timer)
        secondary_cooldown_remaining = self.secondary_projectile_timer
        if secondary_cooldown_remaining > 0:
            secondary_slice_width = self.empty_bar.get_width() * (
                    secondary_cooldown_remaining / self.secondary_cooldown)
            empty_secondary_bar_slice = pygame.Surface((secondary_slice_width, self.empty_bar.get_height()),
                                                       pygame.SRCALPHA)
            empty_secondary_bar_slice.blit(self.empty_bar, (0, 0), (
                self.empty_bar.get_width() - secondary_slice_width, 0, secondary_slice_width,
                self.empty_bar.get_height()))

            # Set the color key for the slice so black is transparent
            empty_secondary_bar_slice.set_colorkey((0, 0, 0))

            screen.blit(empty_secondary_bar_slice,
                        (bar_x_position + self.full_secondary_bar.get_width() - secondary_slice_width,
                         bar_y_position + bar_spacing))

        # Draw the health bar
        screen.blit(self.full_hp_bar, (20, SCREEN_HEIGHT - self.full_hp_bar.get_height() - 20))
        health_lost = self.max_health - int(self.health)

        if health_lost > 0:
            slice_width = self.empty_bar.get_width() * (health_lost / self.max_health)
            empty_bar_slice = pygame.Surface((slice_width, self.empty_bar.get_height()), pygame.SRCALPHA)
            empty_bar_slice.blit(self.empty_bar, (0, 0), (
                self.empty_bar.get_width() - slice_width, 0, slice_width, self.empty_bar.get_height()))

            # Set the color key for the slice so black is transparent
            empty_bar_slice.set_colorkey((0, 0, 0))

            screen.blit(empty_bar_slice,
                        (20 + self.full_hp_bar.get_width() - slice_width,
                         SCREEN_HEIGHT - self.full_hp_bar.get_height() - 20))


class AIPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = [random.random() for _ in range(9)]
        # self.camera = Camera(MAP_WIDTH, MAP_HEIGHT)
        # self.ai_model = self.load_ai_model()

    def load_ai_model(self):
        model = torch.load("path_to_saved_model.pt")
        model.eval()  # set the model to evaluation mode
        return model

    def fire_secondary(self, action, delta_time):
        # keys = self.action_to_keys(action)
        # mouse_buttons, mouse_x, mouse_y = self.action_to_mouse_buttons(action)
        return super().fire_secondary(action, delta_time)

    def action_to_keys(self, action):
        # w, a, s, d, space, _, _, _, _ = action
        w, a, s, d = action # remove space/right click
        # Each movement key is activated if the corresponding action part is greater than 0.5
        keys = {
            "w": w > 0.5,
            "a": a > 0.5,
            "s": s > 0.5,
            "d": d > 0.5,
            "space": False # not controlling space
        }
        return keys

    def action_to_mouse_buttons(self, action):
        x = min(max(0, x * SCREEN_WIDTH), SCREEN_WIDTH)
        y = min(max(0, y * SCREEN_HEIGHT), SCREEN_HEIGHT)

        mouse_buttons = (True, False, False)  # not middle/right clicking
        return mouse_buttons, x, y

    def move(self, action, delta_time):
        # keys = self.action_to_keys(action)
        # mouse_buttons, mouse_x, mouse_y = self.action_to_mouse_buttons(action)
        return super().move(action, delta_time)

    def fire(self, action, delta_time):
        # keys = self.action_to_keys(action)
        # mouse_buttons, mouse_x, mouse_y = self.action_to_mouse_buttons(action)
        return super().fire(action, delta_time)


class Enemy(GameObject):
    def __init__(self, x, y, image, primary_bullet_image, boomerang_bullet_image,
                 attack_list, speed=5, health=100, enemy_path='teleporter', id=0):
        self.random = random.Random(1)
        # def __init__(self, x, y, image, primary_bullet_image, boomerang_bullet_image, speed=5, health=100,
        #              shot_pattern='straight', bullet_path='normal_function', damage=10, bullet_speed=200,
        #              enemy_path='move_toward'):

        # print("at enemy:", bullet_path)
        super().__init__(x, y, image)
        # print(attack_list)  # Should print a list of dictionaries
        self.attack_list = attack_list or [{'shot_pattern': 'straight', 'bullet_path': 'normal_function'
                                               , 'damage': 10, 'bullet_speed': 200, 'bullet_timer': 0,
                                            'bullet_cooldown': 1, 'scale': 5, 'lifetime': 3, ' bullet_image': 'bullet'}] #TODO add bullet encoding
        self.id = id
        self.primary_bullet_image = primary_bullet_image
        self.boomerang_bullet_image = boomerang_bullet_image
        for attack in self.attack_list:
            attack['bullet_timer'] = 2 # give 2 sec for player to respond to enemies position etc
            if attack['bullet_image'] == 'bullet':
                attack['bullet_image'] = self.primary_bullet_image
            elif attack['bullet_image'] == 'fire_ball':
                attack['bullet_image'] = self.boomerang_bullet_image
        # print(self.attack_list)  # Should print a list of dictionaries
        self.health = health
        self.max_health = health
        # self.shot_pattern = shot_pattern

        self.bullets = []
        self.speed = speed * 16.666
        # self.speed = 0 # TODO remove after testing
        # self.damage = damage
        # self.bullet_speed = bullet_speed
        # self.projectile_timer = random.uniform(0, ENEMY_PROJECTILE_COOLDOWN)
        self.float_x = float(self.rect.x)
        self.float_y = float(self.rect.y)
        # self.bullet_path = bullet_path
        self.enemy_path = enemy_path

    def fire(self, player, delta_time):
        dx, dy = calculate_normalized_direction(self.rect, player.rect)
        new_bullets = []
        for attack in self.attack_list:
            # print(attack)
            attack['bullet_timer'] -= delta_time
            if attack['bullet_timer'] <= 0:
                attack['bullet_timer'] = attack['bullet_cooldown']
                if attack['shot_pattern'] == 'spread':
                    new_bullets += self._spread_pattern(dx, dy, attack)
                elif attack['shot_pattern'] == 'spiral':
                    new_bullets += self._spiral_pattern(dx, dy, attack)
                elif attack['shot_pattern'] == 'random':
                    new_bullets += self._random_pattern(dx, dy, attack)
                elif attack['shot_pattern'] == 'boss_swirl_sin':
                    new_bullets += self._boss_swirl_sin(dx, dy, attack)
                elif attack['shot_pattern'] == 'boss_swirl_cos':
                    new_bullets += self._boss_swirl_cos(dx, dy, attack)
                elif attack['shot_pattern'] == 'boss_straight':
                    new_bullets += self._boss_straight(dx, dy, attack)
                else:
                    new_bullets.append(
                        Bullet(self.rect.centerx, self.rect.centery, dx, dy, self.primary_bullet_image,
                               attack=attack))
        return new_bullets

    def _spread_pattern(self, dx, dy, attack):
        bullets = []
        for angle in range(-15, 16, 15):
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, self.primary_bullet_image,
                                  attack=attack))
        return bullets

    def _spiral_pattern(self, dx, dy, attack):
        bullets = []
        for i in range(3):
            angle = i * 360 / 3
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, attack=attack))
        return bullets

    def _random_pattern(self, dx, dy, attack):
        bullets = []
        for _ in range(6): # TODO change back to 8 after testing
            angle = random.randint(0, 360)
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, self.primary_bullet_image,
                                  attack=attack))
        return bullets

    def _boss_swirl_sin(self, dx, dy, attack):
        bullets = []
        for angle in range(-90, 92, 90):
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(
                Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, attack=attack))
        return bullets

    def _boss_swirl_cos(self, dx, dy, attack):
        bullets = []
        for angle in range(-90, 92, 90):
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(
                Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, attack=attack))
        return bullets

    def _boss_straight(self, dx, dy, attack):
        bullets = []
        for angle in range(-120, 121, 80):
            new_dx, new_dy = self._calculate_new_direction(dx, dy, angle)
            bullets.append(
                Bullet(self.rect.centerx, self.rect.centery, new_dx, new_dy, attack=attack))
        return bullets

    @staticmethod
    def _calculate_new_direction(dx, dy, angle):
        new_dx = dx * math.cos(math.radians(angle)) - dy * math.sin(math.radians(angle))
        new_dy = dx * math.sin(math.radians(angle)) + dy * math.cos(math.radians(angle))

        return new_dx, new_dy

    def move(self, target, delta_time):
        if self.enemy_path == 'move_toward':
            dx, dy = self.move_towards(target, delta_time)
            self.float_x += dx * self.speed * delta_time
            self.float_y += dy * self.speed * delta_time

            self.rect.x = int(self.float_x)
            self.rect.y = int(self.float_y)
        # another type of movement (crazy though)
        elif self.enemy_path == 'teleporter':
            if self.random.randint(0, 20) * self.speed / BASE_ENEMY_SPEED < 1:
                self.float_x = min(max(0, self.float_x + self.random.uniform(-SCREEN_WIDTH // 5, SCREEN_WIDTH // 5)),
                                   MAP_WIDTH - -self.rect.width)
                self.float_y = min(max(0, self.float_y + self.random.uniform(-SCREEN_HEIGHT // 5, SCREEN_HEIGHT // 5)),
                                   MAP_HEIGHT - self.rect.height)
                self.rect.x = int(self.float_x)
                self.rect.y = int(self.float_y)

    def move_towards(self, target, delta_time):
        dx = target.rect.centerx - self.rect.centerx
        dy = target.rect.centery - self.rect.centery
        norm = math.sqrt(dx * dx + dy * dy)
        if norm != 0:
            dx = dx / norm
            dy = dy / norm
        # draw_line_between(screen, self.rect, target.rect) # for debugging, only works in top left of screen
        return dx, dy


class Bullet:
    def __init__(self, x, y, dx, dy, bullet_image=None, speed=200, scale=20, damage=10, piercing=False, lifetime=6,
                 creator='player', bullet_path='normal_function', attack=None, homing_factor=0):
        if attack:
            self.damage = attack.get('damage', damage)
            self.speed = attack.get('bullet_speed', speed)
            self.scale = attack.get('scale', scale)
            self.lifetime = attack.get('lifetime', lifetime)
            self.piercing = attack.get('piercing', piercing)
            self.creator = attack.get('creator', creator)
            self.bullet_path = attack.get('bullet_path', bullet_path)
            self.image = attack.get('bullet_image', bullet_image)
            self.homing_factor = attack.get('homing_factor', homing_factor)
            self.id = attack.get('attack_id')

        else:
            # print('no attack')
            self.image = bullet_image
            self.damage = damage
            self.speed = speed
            self.scale = scale
            self.lifetime = lifetime
            self.piercing = piercing
            self.creator = creator
            self.bullet_path = bullet_path
            self.homing_factor = homing_factor

        self.image = pygame.transform.scale(self.image,
                                            (self.image.get_width() * self.scale, self.image.get_height() * self.scale))
        if self.creator == 'enemy' and self.image.get_width() < 18:
            self.image = convert_image_to_grayscale(self.image)
            self.adjust_bullet_color()
        hitbox_size = min(self.image.get_width(), self.image.get_height()) * 0.8  # adjust this value as necessary

        # Adjust x and y so the bullet spawns at the center of the given position
        hitbox_x = x - self.image.get_width() / 2 + (self.image.get_width() - hitbox_size) / 2
        hitbox_y = y - self.image.get_height() / 2 + (self.image.get_height() - hitbox_size) / 2
        self.float_x = hitbox_x
        self.float_y = hitbox_y
        self.rect = pygame.Rect(int(self.float_x), int(self.float_y), hitbox_size, hitbox_size)
        self.dx = dx
        self.dy = dy
        self.base_dx = 0
        self.base_dy = 0
        self.rotated_dx = 0
        self.rotated_dy = 0
        self.hit_targets = set()
        self.time_alive = 0
        self.time_remaining = self.lifetime
        self.angle = -math.degrees(math.atan2(dy, dx))
        # print(bullet_path)
        self.function = getattr(self, self.bullet_path)
        self.update(0)

    def update(self, delta_time, player=None, enemies=None):
        if player or enemies:
            self.homing_angle_adjustment(delta_time, player or enemies)
        self.base_dx, self.base_dy = self.function()

        # Rotate base_dx and base_dy according to bullet's angle
        self.rotated_dx, self.rotated_dy = rotate(self.base_dx, self.base_dy, self.angle)

        self.float_x += self.rotated_dx * self.speed * delta_time
        self.float_y += self.rotated_dy * self.speed * delta_time
        self.rect.x = int(self.float_x)
        self.rect.y = int(self.float_y)
        self.time_remaining -= delta_time
        self.time_alive += delta_time
        if self.time_remaining <= 0:
            # Remove this bullet instance.
            return False
        return True

    def adjust_bullet_color(self, damage_threshold=BASE_ENEMY_DAMAGE, max_damage=BASE_ENEMY_DAMAGE * 3):
        # Calculate the damage ratio
        damage_ratio = max(0, min(1, (self.damage - damage_threshold) / (max_damage - damage_threshold)))

        # Calculate the color steps based on damage_ratio
        red_step = int(abs(212 - 124) * damage_ratio)  # Transition from 212 (golden rod yellow) to 124 (dark brick red)
        green_step = int(abs(148 - 4) * damage_ratio)  # Transition from 148 (golden rod yellow) to 4 (dark brick red)
        blue_step = int(abs(36 - 27) * damage_ratio)  # Transition from 36 (golden rod yellow) to 27 (dark brick red)

        # Base color to add (transition from golden rod yellow to dark brick red)
        red_component = 212 - red_step if 212 > 124 else 212 + red_step
        green_component = 148 - green_step if 148 > 4 else 148 + green_step
        blue_component = 36 - blue_step if 36 > 27 else 36 + blue_step
        alpha_component = 0  # Not changing the alpha component

        # Add the color to the image using BLEND_ADD
        self.image.fill((red_component, green_component, blue_component, alpha_component),
                        special_flags=pygame.BLEND_MIN)

    def sine_wave_function(self):
        frequency = 3
        # print(self.time_alive)
        return 1, 2 * math.sin(self.time_alive * frequency)
        # return 1*random.randint(-2,3), 2* math.sin(time*frequency)

    def player_sine_wave_function(self):
        frequency = 30
        return 1, 2 * math.sin((self.time_alive) * frequency - math.pi / 2.8)

        # return 1*random.randint(-2,3), 2* math.sin(time*frequency)

    def cosine_wave_function(self):
        frequency = 3
        return 1, 2 * math.cos(self.time_alive * frequency)

    def decaying_function(self):
        return 1, math.exp(
            -self.time_alive * 5)  # The bullet will initially move diagonally, then more and more horizontally

    def circular_path_function(self):
        return 3 * math.cos(self.time_alive * 30), 3 * math.sin(
            self.time_alive * 30)  # The bullet will move in a circle

    def spiral_function(self):
        radius = self.time_alive  # radius of the spiral increases as time goes on
        return radius * 5 * math.cos(self.time_alive * 20), radius * 5 * math.sin(self.time_alive * 20)

    def find_closest_target(self, targets):
        closest_target = None
        min_distance_sq = math.inf

        for target in targets:
            # Calculate squared Euclidean distance between self and each target
            distance_sq = (self.rect.centerx - target.rect.centerx) ** 2 + (
                    self.rect.centery - target.rect.centery) ** 2
            if distance_sq < min_distance_sq and target not in self.hit_targets:
                min_distance_sq = distance_sq
                closest_target = target

        return closest_target

    @lru_cache(maxsize=None)
    def atan2_cache(self, y, x):
        return math.atan2(y, x)

    def homing_angle_adjustment(self, delta_time, targets=None):
        if targets:
            closest_target = self.find_closest_target(targets)
            if closest_target:
                dy = closest_target.rect.centery - self.rect.centery
                dx = closest_target.rect.centerx - self.rect.centerx

                bullet_angle = math.atan2(self.rotated_dy, self.rotated_dx)  # can use self.atan2cache
                target_bullet_angle = math.atan2(dy, dx)

                new_bullet_angle = target_bullet_angle - bullet_angle
                # Adjust the angle difference to fall within the -pi to pi range
                if new_bullet_angle > math.pi:
                    new_bullet_angle -= 2 * math.pi
                elif new_bullet_angle < -math.pi:
                    new_bullet_angle += 2 * math.pi

                # make angle adjustment time dependant, account as it is 60fps
                angle_adjustment = 10 * 60 * self.homing_factor * delta_time
                if abs(new_bullet_angle) > 0:
                    if new_bullet_angle < 0:
                        self.angle += min(angle_adjustment, abs(new_bullet_angle * 180 / math.pi))
                    else:
                        self.angle -= min(angle_adjustment, abs(new_bullet_angle * 180 / math.pi))

    def deceleration_function(self):
        return max(0,
                   1 - self.time_alive / 10), 0  # the x-component of the velocity decreases with time, but it won't go below 0

    def quadratic_acceleration_function(self):
        # print(time)
        return min(self.time_alive, 2) ** (2), 0  # the x-component of the velocity increases with the square of time

    def acceleration_function(self):
        return (self.time_alive) ** (1 / 2), 0

    def hard_ramp_acceleration_function(self):
        return min((self.time_alive) ** 40, 2.5), 0

    def normal_function(self):
        return 1, 0

    def draw_debug(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect, 2)  # 2 is the thickness of the line


class EnemyManager:
    def __init__(self, enemy_image, primary_bullet_image, boomerang_bullet_image):
        self.enemies = []
        self.enemy_image = enemy_image
        self.primary_bullet_image = primary_bullet_image
        self.boomerang_bullet_image = boomerang_bullet_image

    def add_enemy(self, x, y, health, speed, attack_list, enemy_path, id):
        new_enemy = Enemy(x, y, self.enemy_image, self.primary_bullet_image,
                          self.boomerang_bullet_image, attack_list, speed, health, enemy_path, id)
        self.enemies.append(new_enemy)

    def update_enemies(self, delta_time, player):
        for enemy in self.enemies:
            enemy.move(player, delta_time)
            if enemy.health <= 0:
                self.enemies.remove(enemy)

    def get_enemies(self):
        return self.enemies

class WaveManager:
    def __init__(self, enemy_manager, difficulty='easy', starting_wave=1):
        self.enemy_manager = enemy_manager
        self.current_wave = starting_wave
        self.wave_cooldown_secs = 3  
        self.difficulty = difficulty
        self.starting_enemies_in_wave = 0
        self.random = random.Random()
        self.difficulty_multipliers = {'easy': 1, 'medium': 1.5, 'hard': 2}
        self.generate_wave_patterns()

    def spawn_location(self, side=None):
        """Generate a spawn location on a given side of the map or randomly."""
        if not side:
            x = self.random.randint(0, MAP_WIDTH)
            y = self.random.randint(0, MAP_HEIGHT)
        else:
            if side == 'top':
                x = self.random.randint(0, MAP_WIDTH)
                y = 0
            elif side == 'bottom':
                x = self.random.randint(0, MAP_WIDTH)
                y = MAP_HEIGHT
            elif side == 'left':
                x = 0
                y = self.random.randint(0, MAP_HEIGHT)
            elif side == 'right':
                x = MAP_WIDTH
                y = self.random.randint(0, MAP_HEIGHT)
        return x, y

    def generate_wave_patterns(self):
        multiplier = self.difficulty_multipliers[self.difficulty]
        wave_patterns = []

        # Enemy and attack IDs
        ENEMY_SPREAD_ID = 1
        ENEMY_SPIRAL_ID = 2
        ENEMY_HOMING_ID = 3
        ENEMY_STRAIGHT_ID = 4
        SPREAD_ATTACK_ID = 1
        SPIRAL_ATTACK_ID = 2
        RANDOM_ATTACK_ID = 3
        STRAIGHT_ATTACK_ID = 4

        # Enemy configurations
        enemy_types = [
            (ENEMY_SPREAD_ID, BASE_ENEMY_SPEED, BASE_ENEMY_HEALTH, [
                {
                    'attack_id': SPREAD_ATTACK_ID,
                    'bullet_cooldown': ENEMY_PROJECTILE_COOLDOWN / 1.5,
                    'bullet_timer': self.random.uniform(0, ENEMY_PROJECTILE_COOLDOWN),
                    'bullet_image': 'bullet',
                    'shot_pattern': 'spread',
                    'bullet_path': 'acceleration_function',
                    'bullet_speed': BASE_ENEMY_BULLET_SPEED * 5,
                    'damage': BASE_ENEMY_DAMAGE,
                    'scale': 5,
                    'lifetime': 1,
                    'creator': 'enemy'
                }
            ]),
            (ENEMY_SPIRAL_ID, BASE_ENEMY_SPEED * 2, BASE_ENEMY_HEALTH * 2, [
                {
                    'attack_id': SPIRAL_ATTACK_ID,
                    'bullet_cooldown': ENEMY_PROJECTILE_COOLDOWN / 5,
                    'bullet_timer': self.random.uniform(0, ENEMY_PROJECTILE_COOLDOWN),
                    'bullet_image': 'fire_ball',
                    'shot_pattern': 'spiral',
                    'bullet_path': 'spiral_function',
                    'bullet_speed': BASE_ENEMY_BULLET_SPEED * 4,
                    'damage': BASE_ENEMY_DAMAGE * 5,
                    'scale': 5,
                    'lifetime': ENEMY_PROJECTILE_COOLDOWN / 3,
                    'creator': 'enemy'
                }
            ]),
            (ENEMY_HOMING_ID, BASE_ENEMY_SPEED * 1.25, BASE_ENEMY_HEALTH * 1.3, [
                {
                    'attack_id': RANDOM_ATTACK_ID,
                    'bullet_cooldown': 5,
                    'bullet_timer': self.random.uniform(0, ENEMY_PROJECTILE_COOLDOWN),
                    'bullet_image': 'bullet',
                    'shot_pattern': 'random',
                    'homing_factor': 0.1,
                    'bullet_path': 'acceleration_function',
                    'bullet_speed': BASE_ENEMY_BULLET_SPEED * 1.3,
                    'damage': BASE_ENEMY_DAMAGE * 1.5 * 3,
                    'scale': 5,
                    'lifetime': 5,
                    'creator': 'enemy'
                }
            ]),
            (ENEMY_STRAIGHT_ID, BASE_ENEMY_SPEED * 0.75, BASE_ENEMY_HEALTH, [
                {
                    'attack_id': STRAIGHT_ATTACK_ID,
                    'bullet_cooldown': ENEMY_PROJECTILE_COOLDOWN * 1.5,
                    'bullet_timer': self.random.uniform(0, ENEMY_PROJECTILE_COOLDOWN),
                    'bullet_image': 'bullet',
                    'shot_pattern': 'straight',
                    'bullet_path': 'acceleration_function',
                    'bullet_speed': BASE_ENEMY_BULLET_SPEED * 4,
                    'damage': BASE_ENEMY_DAMAGE * 5,
                    'scale': 10,
                    'lifetime': 5,
                    'creator': 'enemy'
                }
            ]),
        ]

        # Calculate number of enemies for this wave
        if self.current_wave % 5 == 0:
            number_of_enemies = self.current_wave // 5
        else:
            number_of_enemies = self.current_wave
        self.starting_enemies_in_wave = number_of_enemies
        
        # Generate enemies with configs
        for _ in range(number_of_enemies):
            side = self.random.choice(['top', 'bottom', 'left', 'right'])
            x, y = self.spawn_location(side)
            enemy_type = self.random.choice(enemy_types)
            base_stats, attacks = enemy_type[0:3], enemy_type[3]

            # Scale health and speed based on wave number
            health = base_stats[2] * multiplier * math.log(self.current_wave + 1)
            speed = min(
                PLAYER_SPEED * 0.8,
                base_stats[1] + (PLAYER_SPEED * 1.5 * math.log(self.current_wave + 1)) / (math.log(self.current_wave + 1) + 15)
            )
            enemy_id = base_stats[0]

            # Prepare the enemy data
            enemy_data = {
                'id': enemy_id,
                'x': x,
                'y': y,
                'health': health,
                'speed': speed,
                'attack_list': copy.deepcopy(attacks)
            }
            wave_patterns.append(enemy_data)
            self.enemy_manager.add_enemy(x, y, health, speed, enemy_data['attack_list'], 'move_toward', id=enemy_id)

        self.wave_patterns = wave_patterns

    def start_next_wave(self):
        if self.is_wave_clear() and self.wave_cooldown_secs <= 0:
            self.current_wave += 1
            self.generate_wave_patterns()
            self.wave_cooldown_secs = 3

    def update(self, delta_time):
        if self.wave_cooldown_secs > 0:
            self.wave_cooldown_secs -= delta_time

    def number_of_enemies(self):
        return len(self.enemy_manager.get_enemies())

    def is_wave_clear(self):
        return not bool(self.enemy_manager.get_enemies())

    def get_enemies(self):
        return self.enemy_manager.get_enemies()

    def get_starting_enemies_in_wave(self):
        return self.starting_enemies_in_wave




class Camera:
    def __init__(self, map_width, map_height, screen_width, screen_height):
        self.rect = pygame.Rect(0, 0, screen_width, screen_height)
        self.map_width = map_width
        self.map_height = map_height
        self.screen_width = screen_width
        self.screen_height = screen_height

    def apply(self, target):
        # adjusts the position of a target object based on the camera's position.
        dx, dy = self.rect.topleft
        if isinstance(target, pygame.Rect):
            return target.move(-dx, -dy)
        else:
            return target.rect.move(-dx, -dy)

    def reverse_apply(self, pos, obj=None):
        # calculates world position from screen position.
        dx, dy = self.rect.topleft
        return pos[0] + dx, pos[1] + dy

    def update(self, target):
        # updates the camera's position to center on a target object.
        half_width, half_height = self.screen_width // 2, self.screen_height // 2
        x = target.rect.centerx - half_width
        y = target.rect.centery - half_height

        # clamp position within map boundaries
        x = max(min(x, self.map_width - self.screen_width), 0)
        y = max(min(y, self.map_height - self.screen_height), 0)

        self.rect.topleft = (x, y)

    def in_view(self, target):
        # checks if a target object is within the camera's view.
        return self.rect.colliderect(target.rect)


class UIManager:
    def __init__(self):
        self.buttons = {}

    def add_button(self, name, button):
        self.buttons[name] = button

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for button_name, button in self.buttons.items():
                if button.rect.collidepoint(event.pos):
                    button.callback()

    def draw(self, screen):
        for button_name, button in self.buttons.items():
            if button is not None:
                button.draw(screen)

class Button:
    def __init__(self, rect, text, font, callback):
        self.rect = rect
        self.text = text
        self.font = font
        self.callback = callback

    def collidepoint(self, pos):
        return self.rect.collidepoint(pos)

    def click(self):
        self.callback()

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)
        button_text = self.font.render(self.text, True, (255, 255, 255))
        text_rect = button_text.get_rect(center=self.rect.center)
        screen.blit(button_text, text_rect)


# TODO make clock work for ai when ai is trained
class GameClock:
    def __init__(self, max_ups=60, max_fps=60, max_aps=60, time_source=time.time,
                 update_callback=None, frame_callback=None, ai_callback=None,
                 speed_multiplier=1.5, should_render=True, training_ai=False, playing_ai=False):
        self.time_source = time_source
        self.max_ups = max_ups
        self.max_fps = max_fps
        self.max_aps = max_aps  # max AI updates per second
        self.update_interval = 1.0 / max_ups
        self.frame_interval = 1.0 / max_fps if max_fps > 0 else 0
        self.ai_update_interval = 1.0 / max_aps
        self.last_update_time = self.time_source()
        self.last_frame_time = self.time_source()
        self.last_ai_update_time = self.time_source()
        self.last_counter_reset_time = self.time_source()
        self.num_updates = 0
        self.num_frames = 0
        self.updates_list = []
        self.frames_list = []
        self.ups = 0
        self.fps = 0
        self.aps = 0
        self.update_callback = update_callback
        self.frame_callback = frame_callback
        self.ai_callback = ai_callback
        self.speed_multiplier = speed_multiplier
        self.should_render = should_render
        self.training_ai = training_ai
        self.playing_ai = playing_ai
        self.envs_unwrapped = []
        self.agents = []

    def should_update(self):
        current_time = self.time_source()
        if current_time - self.last_update_time >= self.update_interval / self.speed_multiplier:
            self.num_updates += 1
            self._reset_counters_if_needed(current_time)
            return True
        return False

    def _should_render(self):
        if not self.should_render:
            return False
        current_time = self.time_source()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.num_frames += 1
            self._reset_counters_if_needed(current_time)
            return True
        return False

    def _reset_counters_if_needed(self, current_time):
        if current_time - self.last_counter_reset_time >= 1.0:  # one second
            self.updates_list.append(self.num_updates)
            self.frames_list.append(self.num_frames)
            if len(self.updates_list) > 5:
                self.updates_list.pop(0)
                self.frames_list.pop(0)
            self.ups = sum(self.updates_list) / len(self.updates_list) if self.updates_list else 0
            self.fps = sum(self.frames_list) / len(self.frames_list) if self.frames_list else 0
            self.last_counter_reset_time = current_time
            self.num_updates = 0
            self.num_frames = 0

    def should_update_ai(self):
        current_time = self.time_source()
        if current_time - self.last_ai_update_time >= self.ai_update_interval / self.speed_multiplier:
            # self.last_ai_update_time = current_time
            return True
        return False

    def tick(self, num_logic_updates=1):
        current_time = self.time_source()

        # Logging the current time for debugging
        # print(f'Current Tick Time: {current_time}')

        if self.training_ai:
            for _ in range(num_logic_updates):
                dt_update = self.update_interval * self.speed_multiplier  # use fixed dt for game logic
                self.update_callback(dt_update)
                if self.should_render:
                    self.frame_callback()
        else:
            if self.playing_ai:
                if self.should_update_ai() and self.ai_callback:
                    dt_update = (current_time - self.last_ai_update_time) * self.speed_multiplier

                    # Log the time difference for AI update
                    # print(f"Time since last AI update: {dt_update} secs")

                    self.last_ai_update_time = current_time
                    self.ai_callback(dt_update, self.envs_unwrapped, self.agents)
            if self.should_update() and self.update_callback:
                dt_update = (current_time - self.last_update_time) * self.speed_multiplier

                # Log the time difference for game update
                # print(f"Time since last game update: {dt_update} secs")

                self.last_update_time = current_time
                self.update_callback(dt_update)
            if self._should_render() and self.frame_callback:
                dt_render = current_time - self.last_frame_time

                # Log the time difference for frame render
                # print(f"Time since last frame render: {dt_render} secs")

                self.last_frame_time = current_time
                self.frame_callback()


class Game:
    def __init__(self, rendering=True, num_of_ai_players_at_start=1, training_ai=True, has_human_player=False):
        self.rendering = rendering
        self.has_human_player = has_human_player
        # TODO make training_ai passed in during game creation
        if training_ai:
            # create a separate random instance for enemy spawning with a specific seed
            self.enemy_spawning_random = random.Random(1)  # seed chosen arbitrarily
        self.num_of_ai_players_at_start = num_of_ai_players_at_start
        self.setup_pygame()
        self.load_resources()
        self.game_state = PLAYING
        self.initialize_game_state()
        self.initialize_shop()
        if self.rendering:
            self.setup_ui()

    def initialize_game_state(self):
        self.game_state = PLAYING
        self.all_bullets = []
        self.difficulty_selected = 'easy'
        self.enemy_manager = EnemyManager(self.resources["enemy_image"], self.resources["enemy_primary_bullet_image"],
                                 self.resources["enemy_boomerang_bullet_image"])
        self.wave_manager = WaveManager(self.enemy_manager, self.difficulty_selected, starting_wave=1)
        self.game_objects = self.wave_manager.get_enemies()[:] # make it not the same mutable list




        self.players = []
        player_args = {
            "x": MAP_WIDTH // 2 - self.resources["player_image"].get_width() // 2,
            "y": MAP_HEIGHT // 2 - self.resources["player_image"].get_height() // 2,
            "image": self.resources["player_image"],
            "primary_bullet_image": self.resources["primary_bullet_image"],
            "secondary_bullet_image": self.resources["secondary_bullet_image"],
            "rendering": self.rendering
        }
        if self.rendering:
            player_args.update({
                "full_hp_bar": self.resources["full_hp_bar"],
                "full_dash_bar": self.resources["full_dash_bar"],
                "full_secondary_bar": self.resources["full_secondary_bar"],
                "empty_bar": self.resources["empty_hp_bar"],
                "primary_shot_sound": self.resources["player_shooting_sound"],
            })

        if self.has_human_player:
            self.human_player = Player(**player_args)
            self.players.append(self.human_player)
        for _ in range(self.num_of_ai_players_at_start):  # Use the attribute here
            ai_player = AIPlayer(**player_args)
            self.players.append(ai_player)
        if self.rendering:
            background = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), depth=16)
            fill_background(self.resources["background_tile"], background)
            self.map_obj = GameObject(0, 0, background)
            self.game_objects.insert(0, self.map_obj)
            self.shop_background = pygame.transform.scale(self.resources["shop_background"],
                                                          (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.wave_num_text = self.font.render(f"Wave: {self.wave_manager.current_wave}", 1, (255, 255, 255))
            self.wave_cooldown_text = self.font.render(
                f"Wave: {self.wave_manager.current_wave + 1} Starting in {self.wave_manager.wave_cooldown_secs} Seconds", 1,
                (255, 255, 255))
            if hasattr(self, 'human_player'):
                self.coin_text = self.font.render(f"Coins: {self.human_player.coins}", 1, "white")
            self.fps = self.font.render(f"FPS: {int(self.clock.fps)}", True, (255, 255, 255))
            self.ups = self.font.render(f"UPS: {int(self.clock.ups)}", True, (255, 255, 255))
        self.camera = self.players[0].camera
        self.shop_shown = False
        self.wave_ended = False
        self.delta_time = 0

    def load_resources(self):
        # Resources necessary for gameplay logic and collision detection
        self.resources = {
            "player_image": pygame.image.load('resources/purple_character.png').convert_alpha(),
            "primary_bullet_image": pygame.image.load('resources/ricochet_bullet.png').convert_alpha(),
            "secondary_bullet_image": pygame.image.load('resources/secondary_projectile2.png').convert_alpha(),
            "enemy_primary_bullet_image": pygame.image.load('resources/pistol_bullet.png').convert_alpha(),
            # "enemy_boomerang_bullet_image": pygame.image.load('resources/fire_ball_medium_long.png').convert_alpha(),
            "enemy_boomerang_bullet_image": pygame.image.load('resources/evil_fireball5.png').convert_alpha(),
            "enemy_image": pygame.image.load('resources/enemy.png').convert_alpha(),
        }

        # Resources used only for rendering
        if self.rendering:
            self.resources.update({
                "player_shooting_sound": pygame.mixer.Sound('resources/player_shooting.ogg'),
                "background_tile": pygame.image.load('resources/preview.jpg').convert(),
                "shop_background": pygame.image.load('resources/UI_Flat_Frame_02_Lite.png').convert(),
                "coin": pygame.image.load('resources/Coin.png').convert_alpha(),
                "full_hp_bar": pygame.image.load('resources/full_hp_bar.png').convert_alpha(),
                "full_secondary_bar": pygame.image.load('resources/full_secondary_bar.png').convert_alpha(),
                "full_dash_bar": pygame.image.load('resources/full_dash_bar.png').convert_alpha(),
                "empty_hp_bar": pygame.image.load('resources/empty_hp_bar.png').convert(),
            })

    def setup_pygame(self):
        pygame.init()
        pygame.mixer.init()
        if self.rendering:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), depth=24)
            self.font = pygame.font.SysFont('Arial', 24, bold=True, italic=False)
            self.font_large = pygame.font.SysFont('Arial', 48, bold=True, italic=False)
        else:
            # Set a dummy video mode if not rendering
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.clock = GameClock(update_callback=self.game_logic, frame_callback=self.draw_game_state
                               , ai_callback=ai_callback, should_render=self.rendering)

    def get_game_state(self):
        pass

    def game_logic(self, delta_time):
        self.delta_time = delta_time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                global running
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    if self.game_state == PLAYING:
                        self.game_state = PAUSED
                    elif self.game_state == PAUSED:
                        self.game_state = PLAYING
            if self.rendering:
                # Use the appropriate UIManager based on game state
                ui_manager = self.ui_managers[self.game_state]
                ui_manager.handle_event(event)

        if self.game_state == PLAYING:
            if self.wave_manager.is_wave_clear():
                if not self.wave_ended:
                    self.all_bullets = [bullet for bullet in self.all_bullets if
                                        getattr(bullet, 'creator', None) == 'player']
                    for player in self.players:
                        if isinstance(player, Player):
                            player.coins += 2  # player gets 5 coins after clearing a wave
                    self.wave_ended = True  # Flag to prevent giving coins again for the same wave

                    if self.wave_manager.current_wave % 5 == 0 and self.human_player:
                        self.all_bullets.clear()
                        self.game_state = SHOP
                        return  # Return here to not execute the rest of the code in this iteration

                if self.wave_manager.wave_cooldown_secs <= 0:
                    self.wave_manager.current_wave += 1
                    self.wave_manager.start_next_wave()
                    self.wave_manager.wave_cooldown_secs = 3  # seconds in between each wave
                    self.wave_ended = False  # Reset the flag as the new wave has started

                self.wave_manager.wave_cooldown_secs -= delta_time

            self.update_bullets(self.all_bullets, self.players, self.wave_manager.get_enemies(), delta_time)

            # TODO ai currently controlled by ai env, not game, change later
            # for player in self.players:
            # if isinstance(player, AIPlayer):  # Check if the player is an AI
            #     # Get the current state of the game
            #     state = self.get_game_state()
            #
            #     # Have the AI choose an action based on the state
            #     player.choose_action(state)

            # If the player is a human, get input from the keyboard and mouse
            keys_state = pygame.key.get_pressed()
            keys = {
                "w": keys_state[pygame.K_w],
                "a": keys_state[pygame.K_a],
                "s": keys_state[pygame.K_s],
                "d": keys_state[pygame.K_d],
                "space": keys_state[pygame.K_SPACE]
            }
            mouse_buttons = pygame.mouse.get_pressed()
            action = (keys, mouse_buttons) + pygame.mouse.get_pos()
            if hasattr(self, 'human_player') and self.human_player in self.players:
                # Pass the action to the player's move, fire, and fire_secondary methods
                self.all_bullets.extend(self.human_player.move(action, delta_time))
                self.all_bullets.extend(self.human_player.fire(action, delta_time))
                self.all_bullets.extend(self.human_player.fire_secondary(action, delta_time))

            # TODO maybe make chase and shoot at closest player, updating only maybe ever .25 sec
            for enemy in self.wave_manager.get_enemies():
                if len(self.players) > 1:  # Check if there are any players left
                    chosen_player = random.choice(self.players)
                else:
                    chosen_player = self.players[0] if self.players else None
                if chosen_player:
                    enemy.move(chosen_player, delta_time)
                    self.all_bullets.extend(enemy.fire(chosen_player, delta_time))

            for player in self.players[:]:  # iterate over a copy of the list (removing from mutable while looping might be bad)
                if player.health <= 0:
                    self.players.remove(player)

            # After removing players, check if any players are left
            if not self.players:
                self.game_state = GAME_OVER
                if hasattr(self, 'human_player'):
                    self.draw_game_over()
            else:
                self.camera = self.players[0].camera

    # TODO create shop class
    def initialize_shop(self):
        self.shop_button_positions = {}
        self.shop_button_width = 100

        self.upgrades_grid = [
            ['Damage Up', None, 'Curvy Shots', None, 'Max HP Up'],
            ['Fire Rate Up', None, 'Homing Up', None, 'Bigger Bullets'],
            ['Double Shot', 'Speed Up', None, 'Heal', None],
            ['Piercing Bullets', 'Random Upgrade', 'Secondary Fire Rate Up', None, 'HP On Kill'],
        ]

        self.upgrades = {
            'Heal': {'stock': 100, 'cost': 1, 'callback': self.shop_heal_player},
            'Double Shot': {'stock': 1, 'cost': 4, 'callback': self.shop_double_shot},
            'HP On Kill': {'stock': 2, 'cost': 3, 'callback': self.shop_health_regen},
            'Curvy Shots': {'stock': 1, 'cost': 2, 'callback': self.shop_curvy_shots},
            'Speed Up': {'stock': 3, 'cost': 2, 'callback': self.shop_speed_boost},
            'Damage Up': {'stock': 5, 'cost': 3, 'callback': self.shop_damage_boost},
            'Max HP Up': {'stock': 5, 'cost': 4, 'callback': self.shop_health_upgrade},
            'Fire Rate Up': {'stock': 5, 'cost': 3, 'callback': self.shop_rapid_fire},
            'Bigger Bullets': {'stock': 3, 'cost': 2, 'callback': self.shop_bigger_bullets},
            'Piercing Bullets': {'stock': 1, 'cost': 5, 'callback': self.shop_piercing_bullets},
            'Secondary Fire Rate Up': {'stock': 5, 'cost': 3, 'callback': self.shop_secondary_cooldown},
            'Homing Up': {'stock': 5, 'cost': 2, 'callback': self.shop_homing_upgrade},
            'Random Upgrade': {'stock': 999, 'cost': .001, 'callback': self.random_upgrade},
            # 'Placeholder': {'stock': 0, 'cost': 0, 'callback': self.placeholder_function},
        }

    def draw_game_state(self):
        if self.game_state == PLAYING:
            for player in self.players:
                player.camera.update(player)
            # if hasattr(self, 'human_player') and self.human_player in self.players:
            #     self.camera.update(self.human_player)
            # elif self.players:
            #     self.camera.update(self.players[0])
            self.game_objects = [self.map_obj] + self.wave_manager.get_enemies() + self.players

            self.draw_game_objects(self.game_objects)
            # self.draw_bullets(self.human_player.bullets)
            self.draw_bullets(self.all_bullets)

            if self.wave_manager.wave_cooldown_secs > 0 and self.wave_manager.is_wave_clear():
                if self.clock.num_updates % 6 == 0:
                    self.wave_cooldown_text = self.font.render(
                        f"Wave {self.wave_manager.current_wave + 1} Starting in {round(self.wave_manager.wave_cooldown_secs, 1)} Seconds",
                        1, (255, 255, 255))
                    if hasattr(self, 'human_player'):
                        self.coin_text = self.font.render(f"Coins: {self.human_player.coins}", True, "white")
                if self.clock.num_updates == 1:
                    self.wave_num_text = self.font.render(f"Wave: {self.wave_manager.current_wave + 1}", 1,
                                                          (255, 255, 255))
                self.screen.blit(self.wave_cooldown_text, (20, 20))
            else:
                self.screen.blit(self.wave_num_text, (20, 20))

            if self.clock.num_frames == 1:
                self.fps = self.font.render(f"FPS: {int(self.clock.fps)}", True, (255, 255, 255))
                self.ups = self.font.render(f"UPS: {int(self.clock.ups)}", True, (255, 255, 255))
            if hasattr(self, 'human_player'):
                self.screen.blit(self.coin_text, (20, 45))

            self.screen.blit(self.fps, (SCREEN_WIDTH - 100, 20))
            self.screen.blit(self.ups, (SCREEN_WIDTH - 100, 45))

            # self.player.draw_health_bar(self.screen, 50, SCREEN_HEIGHT - 30, 100, 20)
            if hasattr(self, 'human_player'):
                self.human_player.draw_bars(self.screen)

            # ui_manager = self.ui_managers[self.game_state]
            # ui_manager.draw(self.screen)  # Draw buttons last

        elif self.game_state == GAME_OVER:
            self.screen.fill((0, 0, 0))
            text = self.font.render("Game Over", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)

            ui_manager = self.ui_managers[self.game_state]
            ui_manager.draw(self.screen)  # Draw buttons last

            # ui_manager = self.ui_managers[self.game_state]
            # ui_manager.draw(self.screen)  # Draw buttons last
        elif self.game_state == PAUSED:

            self.pause_logic()
        elif self.game_state == SHOP:
            # ui_manager = self.ui_managers[self.game_state]
            # ui_manager.draw(self.screen)  # Draw buttons last
            self.shop_logic()

        pygame.display.flip()

    # TODO change to players
    def update_bullets(self, bullets, player, enemies, delta_time):
        bullets_to_remove = []
        targets_to_remove = []
        for bullet in bullets:
            if bullet.creator == 'player' and bullet.homing_factor > 0 and enemies:
                is_bullet_alive = bullet.update(delta_time, enemies=enemies)
            elif bullet.creator == 'enemy' and bullet.homing_factor > 0 and player:
                is_bullet_alive = bullet.update(delta_time, player=player)
            else:
                is_bullet_alive = bullet.update(delta_time, )
            if not is_bullet_alive:
                bullets_to_remove.append(bullet)
            if bullet.creator == 'player':
                targets = enemies
            elif bullet.creator == 'enemy':
                targets = player
            for target in targets:
                if bullet.rect.colliderect(target.rect):
                    if hasattr(target, 'dashing') and target.dashing:
                        continue
                    if bullet.piercing:
                        if target not in bullet.hit_targets:
                            bullet.hit_targets.add(target)
                            target.health -= bullet.damage
                    else:
                        bullets_to_remove.append(bullet)
                        target.health -= bullet.damage

                    if target.health <= 0 and target not in targets_to_remove:
                        targets_to_remove.append(target)
                    break

        for bullet in bullets_to_remove:
            if bullet in bullets:
                bullets.remove(bullet)
        for target in targets_to_remove:
            if target in self.wave_manager.get_enemies():
                self.wave_manager.get_enemies().remove(target)
                if hasattr(self, 'human_player'):
                    self.human_player.health = min(self.human_player.max_health,
                                                   self.human_player.health + self.human_player.percent_hp_gain * self.human_player.max_health /
                                                   self.wave_manager.get_starting_enemies_in_wave())
            elif target in targets:
                targets.remove(target)

    def draw_bullets(self, bullets):
        for bullet in bullets:
            if self.camera.in_view(bullet):
                # calculate the current angle based on the current velocity
                current_angle = -math.degrees(math.atan2(bullet.rotated_dy, bullet.rotated_dx))

                bullet_rotated_image = pygame.transform.rotate(bullet.image, current_angle)
                bullet_rotated_rect = bullet_rotated_image.get_rect(center=bullet.rect.center)
                self.screen.blit(bullet_rotated_image, self.camera.apply(bullet_rotated_rect))

    def draw_game_objects(self, objects):
        # Separate the objects by type
        background_objects = [obj for obj in objects if obj is self.map_obj]
        enemy_objects = [obj for obj in objects if isinstance(obj, Enemy)]
        player_objects = [obj for obj in objects if isinstance(obj, Player)]

        # Draw the objects in the desired order
        for obj_list in [background_objects, enemy_objects, player_objects]:
            for obj in obj_list:

                if self.camera.in_view(obj):  # Check if the object is in view
                    self.screen.blit(obj.image, self.camera.apply(obj))
                    if isinstance(obj, (Enemy, AIPlayer)):
                        if obj.health <= 0:
                            objects.remove(obj)
                        else:  # Added
                            pos = self.camera.apply(obj.rect).topleft
                            obj.draw_health_bar(self.screen, *pos, obj.rect.width, 5)  # Added

    def setup_ui(self):
        self.ui_manager = UIManager()

        # Create a separate manager for each game state to avoid overlapping buttons
        self.pause_ui_manager = UIManager()
        self.gameover_ui_manager = UIManager()
        self.shop_ui_manager = UIManager()

        # GAME OVER
        retry_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 50, 100, 50)
        retry_button = Button(rect=retry_button_rect, text='Retry', font=self.font, callback=self.reset)
        self.gameover_ui_manager.add_button('retry_button', retry_button)

        # PAUSE
        resume_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 - 40, 100, 50)
        resume_button = Button(rect=resume_button_rect, text='Resume', font=self.font, callback=self.resume_game)
        self.pause_ui_manager.add_button('resume_button', resume_button)

        quit_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 40, 100, 50)
        quit_button = Button(rect=quit_button_rect, text='Quit', font=self.font, callback=self.quit_game)
        self.pause_ui_manager.add_button('quit_button', quit_button)

        # SHOP
        button_width = 100
        button_height = 50
        margin_x = 40  # increased horizontal space between buttons
        margin_y = 60  # increased vertical space between buttons
        grid_origin_x = SCREEN_WIDTH // 2 - (2.5 * button_width + 2 * margin_x)  # left edge of the grid
        grid_origin_y = SCREEN_HEIGHT // 2 - (2 * button_height + 1.5 * margin_y)  # top edge of the grid

        # Initialize the grid with placeholder buttons
        for i in range(4):  # rows
            for j in range(5):  # columns
                button_x = grid_origin_x + j * (button_width + margin_x)
                button_y = grid_origin_y + i * (button_height + margin_y)
                button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

                upgrade_name = self.upgrades_grid[i][j]  # get upgrade name from grid
                if upgrade_name is not None:  # if there is an upgrade at this position
                    upgrade = self.upgrades[upgrade_name]  # get upgrade info from dictionary
                    button = Button(rect=button_rect, text=upgrade_name, font=self.font, callback=upgrade['callback'])
                    self.shop_ui_manager.add_button(f'button_{i}_{j}', button)

                    button = Button(rect=button_rect, text=upgrade_name, font=self.font, callback=upgrade['callback'])
                    self.shop_ui_manager.add_button(f'button_{i}_{j}', button)
                    self.shop_button_positions[f'button_{i}_{j}'] = (button_x, button_y)  # Store the button position

        # Create the resume button separately
        resume_button_y = grid_origin_y + 4 * (button_height + margin_y)
        resume_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - button_width // 2, resume_button_y, button_width,
                                         button_height)
        resume_shop_button = Button(rect=resume_button_rect, text='Resume', font=self.font, callback=self.resume_game)
        self.shop_ui_manager.add_button('resume_shop_button', resume_shop_button)

        # damage_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 40, 100, 50)
        # quit_button = Button(rect=quit_button_rect, text='Quit', font=self.font, callback=self.quit_game)
        # self.pause_ui_manager.add_button('quit_button', quit_button)

        # add ui_manager to game state
        self.ui_managers = {
            PLAYING: self.ui_manager,
            PAUSED: self.pause_ui_manager,
            GAME_OVER: self.gameover_ui_manager,
            SHOP: self.shop_ui_manager,
        }

    def resume_game(self):
        # print('resuming')
        self.game_state = PLAYING
        # self.clock.resume()

    def quit_game(self):
        pygame.quit()
        sys.exit()

    def pause_logic(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                global running
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    if self.game_state == PLAYING:
                        self.game_state = PAUSED
                        # self.clock.pause()
                    elif self.game_state == PAUSED:
                        self.game_state = PLAYING
                        # self.clock.resume()

            # Use the appropriate UIManager based on game state
            ui_manager = self.ui_managers[self.game_state]
            ui_manager.handle_event(event, self.game_state)

        self.screen.fill((0, 0, 0))
        text = self.font.render("Game Paused", True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        self.screen.blit(text, text_rect)
        ui_manager = self.ui_managers[self.game_state]
        ui_manager.draw(self.screen)  # Draw buttons last
        pygame.display.flip()

    def shop_logic(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.shop_background, (0, 0))
        self.coin_text = self.font_large.render(f"Coins: {self.human_player.coins}", True, "white")
        coin_text_rect = self.coin_text.get_rect(center=(SCREEN_WIDTH // 10, SCREEN_HEIGHT // 12))
        self.screen.blit(self.coin_text, coin_text_rect)
        header_text = self.font_large.render("SHOP", True, (255, 255, 255))
        header_text_rect = header_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 12))
        self.screen.blit(header_text, header_text_rect)
        ui_manager = self.ui_managers[self.game_state]
        ui_manager.draw(self.screen)  # Draw buttons

        for button_name, button in self.shop_ui_manager.buttons.items():
            if button_name not in self.shop_button_positions:  # Skip this button if it's not in the dictionary
                continue
            button_x, button_y = self.shop_button_positions[button_name]  # Retrieve the button position
            upgrade_name = button.text  # Get the name of the upgrade from the button text
            upgrade = self.upgrades[upgrade_name]  # Get the upgrade info from the dictionary

            # Blit the stock text
            stock_text = self.font.render(f"Stock: {upgrade['stock']}", True, "white")
            stock_text_rect = stock_text.get_rect(center=(button_x + self.shop_button_width // 2, button_y - 20))
            self.screen.blit(stock_text, stock_text_rect)

            # Blit the cost text
            cost_text = self.font.render(f"Cost: {upgrade['cost']}", True, "white")
            cost_text_rect = cost_text.get_rect(
                center=(button_x + self.shop_button_width // 2, button_y - 5))  # Adjust position as necessary
            self.screen.blit(cost_text, cost_text_rect)

        # pygame.display.flip()

    def handle_purchase(self, upgrade):
        valid_purchase = False
        if self.human_player.coins >= self.upgrades[upgrade]["cost"] and self.upgrades[upgrade]["stock"] > 0:
            self.human_player.coins -= self.upgrades[upgrade]["cost"]
            self.upgrades[upgrade]["stock"] -= 1
            valid_purchase = True
        return valid_purchase

    def add_coins_stock(self, upgrade):
        self.upgrades[upgrade]["stock"] += 1
        self.human_player.coins += self.upgrades[upgrade]["cost"]

    def random_upgrade(self):
        if self.handle_purchase('Random Upgrade'):
            # Create a list of tuples, each containing an upgrade's name and its callback function
            upgrade_list = [(name, upgrade['callback']) for name, upgrade in self.upgrades.items()]

            # Shuffle the upgrade list
            random.shuffle(upgrade_list)

            # Iterate over the shuffled upgrade list
            for upgrade_name, callback in upgrade_list:
                self.add_coins_stock(upgrade_name)
                # If the purchase is successful, call the callback function
                callback()
                break

    def shop_heal_player(self):
        if self.handle_purchase("Heal"):
            self.human_player.health = self.human_player.max_health

    def shop_double_shot(self):
        if self.handle_purchase("Double Shot"):
            self.human_player.double_shot_active = True

    def shop_health_regen(self):
        if self.handle_purchase("HP On Kill"):
            self.human_player.percent_hp_gain += 0.08

    def shop_curvy_shots(self):
        if self.handle_purchase("Curvy Shots"):
            self.human_player.bullet_path = 'player_sine_wave_function'

    def shop_speed_boost(self):
        if self.handle_purchase("Speed Up"):
            self.human_player.speed *= 1.2

    def shop_damage_boost(self):
        if self.handle_purchase("Damage Up"):
            self.human_player.primary_damage *= 1.2  # Increase damage by 20%

    def shop_health_upgrade(self):
        if self.handle_purchase("Max HP Up"):
            self.human_player.max_health *= 1.15

    def shop_homing_upgrade(self):
        if self.handle_purchase("Homing Up"):
            self.human_player.homing_factor += .2

    def shop_rapid_fire(self):
        if self.handle_purchase("Fire Rate Up"):
            self.human_player.primary_projectile_cooldown *= 0.8  # Decrease cooldown by 20%

    def shop_bigger_bullets(self):
        if self.handle_purchase("Bigger Bullets"):
            self.human_player.primary_bullet_size += 1

    def shop_piercing_bullets(self):
        if self.handle_purchase("Piercing Bullets"):
            self.human_player.primary_bullet_piercing = True

    def shop_secondary_cooldown(self):
        if self.handle_purchase("Secondary Fire Rate Up"):
            self.human_player.secondary_cooldown *= .7

    def draw_game_over(self):
        self.screen.fill((0, 0, 0))
        text = self.font.render("Game Over", True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)
        self.ui_manager.draw(self.screen)
        # pygame.display.flip()

    def reset(self):
        self.setup_pygame()
        # self.load_resources()
        self.game_state = PLAYING
        self.initialize_game_state()
        # self.initialize_shop()
        if self.rendering:
            self.setup_ui()


def ai_callback(dt, envs_unwrapped, agents):
    for env_unwrapped, agent in zip(envs_unwrapped, agents):
        if env_unwrapped.player in env_unwrapped.game.players:  # Only take action if the player is still in the game
            env_unwrapped.player.delta_time = dt  # Update the player's delta_time before taking action
            action, _ = agent.predict(env_unwrapped.get_state())
            obs, reward, done, _, info = env_unwrapped.step(action)


def calculate_normalized_direction(source_pos, target_pos):
    target_x = getattr(target_pos, 'centerx', target_pos[0])
    target_y = getattr(target_pos, 'centery', target_pos[1])

    source_x = getattr(source_pos, 'centerx', source_pos[0])
    source_y = getattr(source_pos, 'centery', source_pos[1])

    dx, dy = target_x - source_x, target_y - source_y

    norm = math.sqrt(dx * dx + dy * dy)
    if norm == 0:
        dx = 0
        dy = -1
    else:
        dx, dy = dx / norm, dy / norm
    return dx, dy


def draw_debugging_lines(screen):
    red = (255, 0, 0)
    green = (0, 255, 0)
    pygame.draw.line(screen, red, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT), 1)
    pygame.draw.line(screen, green, (0, SCREEN_HEIGHT // 2), (SCREEN_WIDTH, SCREEN_HEIGHT // 2), 1)


def convert_image_to_grayscale(image):
    # Get a copy of the image
    grayscale_image = image.copy()

    # Get the pixel array of the image
    pixel_array = pygame.PixelArray(grayscale_image)

    # Iterate over each pixel in the pixel array
    for y in range(image.get_height()):
        for x in range(image.get_width()):
            # Get the color of the pixel
            r, g, b, a = image.get_at((x, y))

            # Convert the color to grayscale
            gray = int(0.3 * r + 0.59 * g + 0.11 * b)

            # Set the new color of the pixel
            pixel_array[x, y] = (gray, gray, gray, a)

    # Delete the pixel array to make the changes take effect
    del pixel_array

    return grayscale_image


def fill_background(background_tile, background):
    for i in range(0, MAP_WIDTH, background_tile.get_width()):
        for j in range(0, MAP_HEIGHT, background_tile.get_height()):
            background.blit(background_tile, (i, j))


def rotate(dx, dy, angle):
    rad = math.radians(angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    return dx * cos + dy * sin, dy * cos - dx * sin



