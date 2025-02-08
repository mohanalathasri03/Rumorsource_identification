-- Drop the database if it exists
DROP DATABASE IF EXISTS rumor;

-- Create the database
CREATE DATABASE IF NOT EXISTS rumor;

-- Switch to the created database
USE rumor;

-- Create the user table
CREATE TABLE `users` (
    `name` VARCHAR(225) NOT NULL,
    `email` VARCHAR(225) NOT NULL,
    `phonenumber` VARCHAR(225) NOT NULL,
    `password` VARCHAR(225) NOT NULL,
    `age` VARCHAR(225) NOT NULL,
    `place` VARCHAR(225) NOT NULL
);
