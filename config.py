# config.py

# Screen settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Player settings
PLAYER_SPEED = 20
PLAYER_DAMAGE = 2
MAX_PLAYER_HEALTH = 250

# Projectile settings
PROJECTILE_COOLDOWN = 0.2  # in seconds
SECONDARY_PROJECTILE_COOLDOWN = 1  # in seconds
MAX_BULLET_SPEED = 1000

# Enemy settings
ENEMY_PROJECTILE_COOLDOWN = 0.8  # in seconds
BASE_ENEMY_DAMAGE = 10
BASE_ENEMY_BULLET_SPEED = 225
BASE_ENEMY_HEALTH = 100
BASE_ENEMY_SPEED = 500
MAX_ENEMY_SPEED = 500

# Map settings
MAP_WIDTH = 1000  # for training
MAP_HEIGHT = 800

# Game states
PLAYING = 1
GAME_OVER = 2
PAUSED = 3
SHOP = 4
