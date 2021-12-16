{-# LANGUAGE OverloadedStrings #-}

module PeulScraper
    ( getProblem
    ) where

import Data.Text ( Text )
import Text.HTML.TagSoup ( Tag )
import Text.HTML.Scalpel

getProblem :: Int -> IO [Tag Text]
getProblem n = fetchTags ("https://projecteuler.net/minimal=" ++ show n)

getProblemIO :: IO ()
getProblemIO = do
    n <- read <$> getLine :: IO Int
    p <- getProblem n
    mapM_ print p
