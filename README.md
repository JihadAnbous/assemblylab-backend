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
`docker-compose exec web python manage.py makemigrations accounts applications attachments components customers products vendors`  
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
