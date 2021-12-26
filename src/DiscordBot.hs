{-# LANGUAGE OverloadedStrings #-}

module DiscordBot ( runBot ) where

import Control.Monad ( forM_, void, unless )

import qualified Data.Text as T
import qualified Data.Text.IO as TIO

import UnliftIO ( liftIO )
import UnliftIO.Concurrent

import Discord
import Discord.Types
import qualified Discord.Requests as R

import CommandParsing

runBot :: IO ()
runBot = do
    token <- TIO.readFile "./src/auth_token"
    let opts = def { discordToken = token
                   --, discordOnStart = startHandler
                   , discordOnEnd = liftIO $ putStrLn "Ended"
                   , discordOnEvent = eventHandler
                   , discordOnLog = \s -> TIO.putStrLn s >> TIO.putStrLn ""
                   }
    void $ runDiscord opts
    return ()

startHandler :: DiscordHandler ()
startHandler = do
    Right partialGuilds <- restCall R.GetCurrentUserGuilds
    forM_ partialGuilds $ \pg -> do
        Right guild <- restCall $ R.GetGuild $ partialGuildId pg
        Right chans <- restCall $ R.GetGuildChannels $ guildId guild
        let isAppr ch = isTextChannel ch && channelName ch == "peul"
        forM_ (filter isAppr chans) $ \ch ->
            void $ restCall $ R.CreateMessage (channelId ch) "Hello, World!"

isTextChannel :: Channel -> Bool
isTextChannel (ChannelText {}) = True
isTextChannel _ = False

eventHandler :: Event -> DiscordHandler ()
eventHandler event = case event of
    MessageCreate msg -> unless (userIsBot $ messageAuthor msg) $
        case parseCommand $ messageText msg of
            Just (ViewProblem n) -> do
                void $ restCall $ R.CreateMessage (messageChannel msg) $ 
                    T.pack $ "https://projecteuler.net/problem=" ++ show n
                threadDelay $ 2 * 10^6
            Just (CommandError err) -> do
                void $ restCall $ R.CreateMessage (messageChannel msg) err
                threadDelay $ 2 * 10^6
            Nothing -> return ()
    _ -> return ()

