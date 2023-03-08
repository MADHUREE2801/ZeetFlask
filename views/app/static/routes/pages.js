const express=require("express");
const router=express.Router();
const userController=require('../controllers/users');

router.get("/login",(req,res)=>{
    // res.send("<h1>Hello Sai hi</h1>");
    res.render('login');
 });
 
 router.get("/signup",(req,res)=>{
     // res.send("<h1>Hello Sai hi</h1>");
     res.render('signup');
  });
  router.get("/",(req,res)=>{
     res.render('home');
  });
  router.get("/welcome",(req,res)=>{
     res.render('welcome');
  });
 
 module.exports=router;