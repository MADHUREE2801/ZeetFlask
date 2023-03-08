const mysql=require('mysql');
const bcrypt=require('bcryptjs');
const jwt=require("jsonwebtoken");
const {promisify}=require('util');

const db=mysql.createConnection({
    host:process.env.DATABASE_HOST,
    user:process.env.DATABASE_USER,
    password:process.env.DATABASE_PASS,
    database:process.env.DATABASE,
    });
    
    exports.login=async(req,res)=>{
   
        try{
            
            const {email,password}=req.body;
            if(!email || !password){
             return res.status(400).render('login',{msg:'Please Enter Your Email and Password',msg_type:"error"});
            }
            db.query('select * from users where email=?',[email],async(error,result)=>{
                console.log(result);
                if(result.length<=0)
                return res.status(401).render('login',{msg:'Not a Valid User',msg_type:"error"});
                
                else{
                    
                    if(!(await bcrypt.compare(password,result[0].pass)))
                    {
                        return res.status(401).render('login',{msg:'Incorrect Password',msg_type:"error"});

                    }
                    else
                    {
                       /* const id=result[0].ID;
                        const token=jwt.sign({id:id},process.env.JWT_SECRET,{expiresIn:process.env.JWT_EXPIRES_IN,
                        });

                        console.log(token);
                        const cookieoptions={
                            expires:newDate(Date.now()+process.env.JWT_COOKIE_EXPIRES*24*60*60*1000),
                            httpOnly:true
                        };
                        res.cookie("sai",token,cookieoptions);*/
                         res.status(200).render('welcome');
                    }
                }
            });
        } 
        catch(error)
        {
            console.log(error);
        }

    };


exports.register=(req,res)=>{
    console.log(req.body);
    /*
    const name=req.body.fullname;
    const email=req.body.email;
    const password=req.body.password;
    const confirm_password=req.body.confirm_password;*/
    //res.send("Form submitted");

    const {fullname,email,password,confirm_password}=req.body;
    db.query('select email from users where email=?',[email],async(error,result)=>{
        if(error)
        {
            console.log(error);
        }

        if(result.length>0)
        {
            return res.render('signup',{msg:'Email id already taken',msg_type:"error"});
        }

        else if(password!==confirm_password)       //== checks case
        {
            return res.render('signup',{msg:'Password does not match',msg_type:"error"});

        }

        let hashedpassword=await bcrypt.hash(password,8);

        db.query(
        'insert into users  set  ?',
        {fullname:fullname,email:email,pass:hashedpassword},
        (error,result)=>{

            if(error)
            {
                console.log(error);
            }
            else{
                console.log(result);
                return res.render('signup',{msg:'User Registration Success',msg_type:'good'});
            }
        });
        //console.log(hashedpassword);
    });
    //console.log(fullname);
    //console.log(email);

};


/*exports.isLoggedIn=async(req,res,next)=>{
    if(req.cookies.sai)
    {
        try{
            const decode=await promisify(jwt.verify)(
            req.cookies.sai,
            process.env.JWT_SECRET
            );
            db.query("select * from users where id=?",
            [decode.id],
            (err,results)=>{
                if(!results)
                {
                    return next();
                }
                req.user=results[0];
                return next();
            }
            );
        }
        catch(error){
         console.log(error);
         return next();
        }
     } else{
            next();
        }
    };
*/