# Introduction
This project is designed to allow users to create 2D component templates comprised of points via coordinates. 

For example, a user may create a "Quadrilateral" template (component template) with points A, B, C & D. Users would then use this template to create multiple quadrilaterals (components) by simply giving the component a name and each point some x and y coordinates. A point may be geometrical (pins or joints) or observational (centres of gravity).

It should then allow the user to create an assembly via fixing a component and adding coincident, distance and angle constraints between 2 or more components. Upon creating such an assembly, the user will also be able to create "studies" - whereby variable constraints such as a variable distance constraint (minimum distance, maximum distance & distance increment) or variable angle constraint (minimum angle, maximum angle & angle increment) can be added which would allow the user to loop the assembly through multiple different positions.

The most powerful feature is the "References" feature. A reference allows a user to monitor a dimension, coordinate or position. After allowing the assembly to solve for every position/increment, a table is generated that displays the results of every reference at every respective variable constraint increment.

This software should come in handy for engineers that need to perform multiple kinematic studies for complex assemblies that may be simplified in 2D space - especially simulations where centres of gravity are crucial.



# Docker

Create an image:  
`docker-compose up -d --build`  
Run a container:  
`docker-compose up`  
Run a container in detached mode:  
`docker-compose up -d`  
Check logs:  
`docker-compose logs`  
Stop a container:  
`docker-compose down`

# Django commands

Create migration files:  
`docker-compose exec web python manage.py makemigrations accounts assemblies components constraints references`  
Apply migration files to database:  
`docker-compose exec web python manage.py migrate`  
Create admin user:  
`docker-compose exec web python manage.py createsuperuser`  
Restart database:  
`docker-compose restart db`  
For major database schema changes, delete all migration folders in every application and then recreate migration files.

### Extract SQL code from a migration file

```
docker-compose exec web python manage.py sqlmigrate app_name 0001
```

### Extract SQL code from a migration file into a separate .sql file

```
docker-compose exec web python manage.py sqlmigrate app_name 0001 > app_name_migration.sql
```

### Test run

```
 python -m django_project.utils.jax_examples
```
