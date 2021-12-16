{-# LANGUAGE OverloadedStrings #-}

module CommandParsing ( parseCommand, Command(..) ) where

import qualified Data.Text as T
import Data.Text.Read ( decimal )

data Command
    = ViewProblem Int
    | CommandError T.Text

parseCommand :: T.Text -> Maybe Command
parseCommand msg
    | T.head msg == ',' = Just $ case T.words $ T.tail msg of
        "view" : fields -> parseView fields
        _               -> CommandError "Unknown command"
    | otherwise = Nothing

parseView :: [T.Text] -> Command
parseView [] = CommandError "Missing field"
parseView (t:_)
    | t == "random" = CommandError "Random not yet implemented"
    | otherwise = case toNumber t of
        Nothing -> CommandError "Not a number"
        Just n  -> ViewProblem n
    where toNumber :: T.Text -> Maybe Int
          toNumber t = case decimal t of { Right (n,"") -> Just n ; _ -> Nothing }
