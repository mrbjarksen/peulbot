{-# LANGUAGE OverloadedStrings #-}

module PeulScraper where

import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Text.HTML.Scalpel

import Network.HTTP.Req

data SignInData = SignInData
    { username :: T.Text
    , password :: T.Text
    , captcha :: T.Text
    , csrf_token :: T.Text
    } deriving (Show, Generic)

instance ToJSON SignInData
instance FromJSON SignInData

data AccountData = AccountData
    { username :: T.Text
    , password :: T.Text
    , friend_code :: T.Text
    , recovery_key :: T.Text
    } deriving (Show, Generic)

instance ToJSON AccountData
instance FromJSON AccountData

getSessionId :: IO CookieJar
getSessionId = runReq defaultHttpConfig $ do
    info <- req GET (https "projecteuler.net" /: "sign_in") NoReqBody jsonResponse $
        queryFlag "csrf_token" <> queryFlag "captcha_audio"
    accountData <- fromJSON "./account_info" :: AccountData
    let signInData = SignInData
        { username   = accountData.username
        , password   = acciuntData.password
        , captcha    = solveCaptcha info.captcha_audio
        , csrf_token = info.csrf_token
        }
    response <- req POST (https "projecteuler.net" /: "sign_in")
        (ReqBodyJson signInData) jsonResponse mempty
    return $ responseCookieJar response

getHistory :: T.Text -> IO [(Int, Datetime)]
getHistory user = do
    sessId <- getSessionId
    historyPage <- runReq defaultHttpConfig $ 
        req GET (https "projecteuler.net" /: "user=" /: user /: ";show=history")
            NoReqBody bsResponse $ 
            cookieJar sessId
    return $ scrapeHistory historyPage

scrapeHistory :: ByteString -> [(Int, DateTime)]
scrapeHistory page = 
