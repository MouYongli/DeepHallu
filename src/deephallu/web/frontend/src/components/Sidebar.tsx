"use client"

import { useState } from "react"
import {
  Folder,
  FolderOpen,
  Image,
  Upload,
  FileText,
  ChevronRight,
  ChevronDown,
  X
} from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@radix-ui/react-collapsible"
import { Separator } from "@radix-ui/react-separator"
import { cn } from "@/lib/utils"
import { mockImages, benchmarkCategories } from "@/lib/mockData"

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onSelectImage?: (imageId: string) => void
}

interface FileItem {
  name: string
  type: 'file' | 'folder'
  path: string
  children?: FileItem[]
  id?: string
  size?: string
  uploadTime?: Date
}

// Generate file structure from mock data
const generateFileStructure = (): FileItem[] => {
  const mmeCategories: { [key: string]: FileItem[] } = {}
  const llavaFiles: FileItem[] = []

  mockImages.forEach(image => {
    if (image.dataset === 'MME') {
      if (!mmeCategories[image.category]) {
        mmeCategories[image.category] = []
      }
      mmeCategories[image.category].push({
        name: image.name,
        type: 'file',
        path: image.path,
        id: image.id
      })
    } else if (image.dataset === 'LLaVA Bench') {
      llavaFiles.push({
        name: image.name,
        type: 'file',
        path: image.path,
        id: image.id
      })
    }
  })

  const mmeChildren = Object.entries(mmeCategories).map(([category, files]) => ({
    name: category,
    type: 'folder' as const,
    path: `/dataset/mme_images/${category.toLowerCase().replace(' ', '_')}`,
    children: files
  }))

  return [
    {
      name: "MME Dataset",
      type: "folder",
      path: "/dataset/mme",
      children: mmeChildren
    },
    {
      name: "LLaVA Bench",
      type: "folder",
      path: "/dataset/llava_bench",
      children: llavaFiles
    },
    {
      name: "Models",
      type: "folder",
      path: "/models",
      children: [
        { name: "llava-next-7b", type: "folder", path: "/models/llava-next-7b" },
        { name: "llava-next-13b", type: "folder", path: "/models/llava-next-13b" },
        { name: "llava-next-34b", type: "folder", path: "/models/llava-next-34b" },
      ]
    }
  ]
}

function FileTreeItem({ item, level = 0, onSelectImage }: {
  item: FileItem;
  level?: number;
  onSelectImage?: (imageId: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false)
  const hasChildren = item.children && item.children.length > 0

  const handleClick = () => {
    if (item.type === 'file' && item.id && onSelectImage) {
      onSelectImage(item.id)
    }
  }

  return (
    <div>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <div
            className={cn(
              "flex items-center space-x-2 px-2 py-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded cursor-pointer text-sm",
              "transition-colors"
            )}
            style={{ paddingLeft: `${level * 16 + 8}px` }}
            onClick={item.type === 'file' ? handleClick : undefined}
          >
            {hasChildren && (
              <>
                {isOpen ? (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-gray-500" />
                )}
              </>
            )}
            {!hasChildren && <div className="w-4" />}

            {item.type === 'folder' ? (
              isOpen ? (
                <FolderOpen className="h-4 w-4 text-blue-500" />
              ) : (
                <Folder className="h-4 w-4 text-blue-500" />
              )
            ) : (
              <Image className="h-4 w-4 text-green-500" />
            )}

            <span className="text-gray-700 dark:text-gray-300 truncate">
              {item.name}
            </span>
          </div>
        </CollapsibleTrigger>

        {hasChildren && (
          <CollapsibleContent>
            {item.children?.map((child, index) => (
              <FileTreeItem
                key={index}
                item={child}
                level={level + 1}
                onSelectImage={onSelectImage}
              />
            ))}
          </CollapsibleContent>
        )}
      </Collapsible>
    </div>
  )
}

export function Sidebar({ isOpen, onClose, onSelectImage }: SidebarProps) {
  const [dragOver, setDragOver] = useState(false)
  const [uploadedImages, setUploadedImages] = useState<FileItem[]>([])
  const [uploading, setUploading] = useState(false)
  const fileStructure = generateFileStructure()

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    handleFiles(files)
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    handleFiles(files)
    // Reset input value to allow re-uploading the same file
    e.target.value = ''
  }

  const handleFiles = async (files: File[]) => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'))

    if (imageFiles.length === 0) {
      alert('Please upload only image files (JPG, PNG, GIF, etc.)')
      return
    }

    setUploading(true)

    try {
      const newImages: FileItem[] = []

      for (const file of imageFiles) {
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
          alert(`File ${file.name} is too large. Maximum size is 10MB.`)
          continue
        }

        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file)

        // Generate unique ID
        const id = `uploaded_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

        const newImage: FileItem = {
          id,
          name: file.name,
          type: 'file',
          path: objectUrl,
          size: formatFileSize(file.size),
          uploadTime: new Date()
        }

        newImages.push(newImage)

        // Add to mock images for selection
        mockImages.push({
          id,
          name: file.name,
          path: objectUrl,
          category: 'Uploaded',
          dataset: 'User Upload',
          resolution: 'Unknown',
          format: file.name.split('.').pop()?.toUpperCase() || 'Unknown',
          size: formatFileSize(file.size),
          description: `User uploaded image: ${file.name}`
        })
      }

      setUploadedImages(prev => [...prev, ...newImages])

      if (newImages.length === 1 && onSelectImage) {
        // Auto-select if only one image uploaded
        onSelectImage(newImages[0].id!)
      }

    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  if (!isOpen) return null

  return (
    <div className="w-80 h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      {/* Sidebar Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          File Explorer
        </h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
        >
          <X className="h-4 w-4 text-gray-500" />
        </button>
      </div>

      {/* Upload Area */}
      <div className="p-4">
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-6 text-center transition-colors",
            dragOver
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
              : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500",
            uploading && "opacity-50 pointer-events-none"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploading ? (
            <>
              <div className="animate-spin h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-sm text-blue-600 dark:text-blue-400 mb-2">
                Uploading images...
              </p>
            </>
          ) : (
            <>
              <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                Drop images here or
              </p>
              <label htmlFor="file-upload" className="cursor-pointer">
                <span className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                  browse files
                </span>
                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  multiple
                  accept="image/*"
                  onChange={handleFileUpload}
                />
              </label>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                PNG, JPG, GIF up to 10MB
              </p>
            </>
          )}
        </div>
      </div>

      <Separator className="bg-gray-200 dark:bg-gray-700 h-px" />

      {/* File Tree */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {/* Uploaded Images Section */}
          {uploadedImages.length > 0 && (
            <div className="mb-4">
              <FileTreeItem
                item={{
                  name: `Uploaded Images (${uploadedImages.length})`,
                  type: 'folder',
                  path: '/uploaded',
                  children: uploadedImages
                }}
                onSelectImage={onSelectImage}
              />
            </div>
          )}

          {/* Dataset Structure */}
          {fileStructure.map((item, index) => (
            <FileTreeItem
              key={index}
              item={item}
              onSelectImage={onSelectImage}
            />
          ))}
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Ready for analysis</span>
          </div>
          {uploadedImages.length > 0 && (
            <div className="flex items-center space-x-1">
              <Upload className="h-3 w-3" />
              <span>{uploadedImages.length} uploaded</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}